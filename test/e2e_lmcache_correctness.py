#!/usr/bin/env python3
"""E2E correctness gate for LMCache GPU-prefill integration (docs/local/
LMCACHE_INTEGRATION_PLAN.md §4 step 4).

This is NOT a "did it crash" check -- a prior LMCache release crashed on
exactly this save path (MultiConnector's wait_for_save() -> LMCacheConnectorV1
-> multi_layer_kv_transfer -> NotImplementedError for the HND KV-cache layout
NixlConnector requires). The version pinned since (see config/versions.env's
LMCACHE_VERSION comment) avoids that crash, but predates LMCache's HND kernel
work entirely -- so the real open question is whether it handles our
forced-HND tensors *correctly* via some generic path, or silently mishandles
them. A silent-corruption bug would look like occasional, easy-to-dismiss
output drift, not a crash -- so this test's bar is token-for-token identical
output between a cold (no-cache) response and a cache-hit replay, repeated
multiple times, not just "a hit was reported."

IMPORTANT, established empirically (docs/local/LMCACHE_INTEGRATION_PLAN.md
§6): this stack has a real, LMCache-INDEPENDENT floating-point noise floor --
even a confirmed genuine vLLM-native GPU-resident cache hit (zero LMCache
involvement possible) occasionally produces a different completion on replay,
from ordinary GPU non-associativity. A same-N control measured 0-1
mismatches per 10-15 trials. This test therefore does NOT require 100% of
trials to match -- see --max-mismatches.

Assumes control-plane + >=1 GPU prefill worker + RDU decode are already
running (same convention as test/e2e_kv_routing.py).

Usage:
  python3 test/e2e_lmcache_correctness.py
  python3 test/e2e_lmcache_correctness.py --trials 15 --filler-count 15

Exit code: 0 = correctness gate passes, 1 = a trial failed, 2 = setup error.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import urllib.request
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_kv_routing import load_env  # noqa: E402 -- shared env-file parser, not duplicated


def discover_logs(n_workers):
    """Find the most recent gpu_prefill_*.log group (same launch timestamp).

    Requires the latest-timestamped group to actually have >= n_workers logs
    (matches test/e2e_kv_routing.py's own check) -- otherwise a stale/partial
    log group (e.g. a crashed single-worker retry) with a later timestamp
    than the real current launch could get silently selected, causing
    lmcache_hit_tokens_for() to only ever grep one worker's log.
    """
    pattern = os.path.join(REPO_ROOT, "logs", "*_gpu_prefill_*.log")
    files = glob.glob(pattern)
    groups = defaultdict(dict)
    for f in files:
        m = re.search(r"(\d{8}_\d{6})_gpu_prefill_(\d+)\.log$", f)
        if not m:
            continue
        ts, idx = m.group(1), int(m.group(2))
        groups[ts][idx] = f
    if not groups:
        raise SystemExit("No logs/*_gpu_prefill_*.log found -- is a GPU prefill worker running?")
    latest_ts = max(groups)
    group = groups[latest_ts]
    if len(group) < n_workers:
        raise SystemExit(f"Most recent launch ({latest_ts}) only has {len(group)} worker log(s), "
                          f"need >= {n_workers}. Pass --logs explicitly if this is stale.")
    return [group[i] for i in sorted(group)]


_CORPUS_WORDS = None
_CORPUS_PATH = os.path.join(REPO_ROOT, "test", "fixtures", "pride_and_prejudice.txt")


def _corpus_words():
    """Real, coherent natural-language text (Pride and Prejudice, public
    domain, via Project Gutenberg -- test/fixtures/pride_and_prejudice.txt,
    tracked in git), split into words. Deliberately NOT random dictionary
    words or "tokNNNNNN"-style pseudo-random tokens (used by
    test/e2e_kv_routing.py, fine there since that test only checks a
    cached_tokens count, never decoded text) -- greedy decoding from
    out-of-distribution/gibberish token streams tends to fall into degenerate
    repeat-loops that are hypersensitive to ordinary GPU floating-point
    non-associativity. Real in-distribution text avoids that chaotic decoding
    regime, giving a much fairer test of whether LMCache's own KV round-trip
    is correct.
    """
    global _CORPUS_WORDS
    if _CORPUS_WORDS is None:
        if not os.path.exists(_CORPUS_PATH):
            raise SystemExit(f"Missing test corpus: {_CORPUS_PATH}\n"
                              "This should be tracked in git (test/fixtures/) -- if it's "
                              "missing, the checkout is incomplete.")
        with open(_CORPUS_PATH, errors="replace") as f:
            text = f.read()
        _CORPUS_WORDS = text.split()
    return _CORPUS_WORDS


def make_prompt(seed, n_words):
    """A random contiguous window of real corpus text, n_words long."""
    words = _corpus_words()
    rng = random.Random(seed)
    start = rng.randint(0, max(1, len(words) - n_words - 1))
    return " ".join(words[start:start + n_words])


def send(endpoint, model, prompt, max_tokens=2):
    """Deterministic (temperature=0) completion request. Returns parsed JSON.

    max_tokens defaults deliberately short: this stack has real,
    LMCache-independent run-to-run floating-point drift that compounds over
    a long autoregressive generation -- even a confirmed genuine vLLM-native
    GPU-resident cache hit (zero LMCache involvement) mismatched 9/10 times
    at 64 tokens, but only 0-1/10 at max_tokens<=2. A short window is the
    only way to get a clean signal for "did LMCache's KV round-trip preserve
    correctness" that isn't swamped by this stack's own baseline noise floor.
    """
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "seed": 0}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


_HIT_LINE_RE = re.compile(
    r"Reqid:\s*([0-9a-fA-F-]+).*?Total tokens\s+(\d+).*?"
    r"Inference Engine computed tokens:\s*(\d+).*?"
    r"LMCache hit tokens:\s*(\d+)"
)


def lmcache_hit_tokens_for(req_id, log_contents_by_path):
    """Grep every worker log for LMCache's own per-request hit-token line.

    Real line (lmcache/integration/vllm/vllm_v1_adapter.py, get_num_new_matched_tokens):
      "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d,
       LMCache hit tokens: %d, need to load: %d"

    This is the ground-truth discriminator between "vLLM's own native GPU
    prefix cache already had it" (LMCache hit tokens would be 0 or small --
    nothing left for LMCache to supply) and "LMCache's CPU tier actually
    served a meaningful chunk of this request" (a large LMCache hit tokens
    value). Match is by substring containment of the OpenAI-facing completion
    id inside LMCache's own (differently-suffixed) internal Reqid field --
    confirmed correct by direct log inspection (LMCache appends its own
    suffix to the client-visible UUID), not a loose/accidental match. If a
    future vLLM/LMCache version ever changes this id relationship, every
    trial would start returning None here with no other symptom -- if this
    function silently goes quiet across an entire run (0 matches on every
    trial) where hits are otherwise expected, suspect this correlation first,
    not just "request too short."

    Returns (engine_computed, lmcache_hit, total, path) for the first match
    found across all worker logs, or None if no matching line is found at all.
    """
    for path, content in log_contents_by_path.items():
        for m in _HIT_LINE_RE.finditer(content):
            if req_id and req_id in m.group(1):
                total, engine_computed, lmcache_hit = int(m.group(2)), int(m.group(3)), int(m.group(4))
                return engine_computed, lmcache_hit, total, path
    return None


class _LogTailReader:
    """Reads each worker log incrementally (only new bytes since the last
    call), keeping the running content in memory -- avoids re-reading the
    entire, ever-growing log file from byte 0 on every trial (which scales
    ~quadratically with trial count for long runs)."""

    def __init__(self, paths):
        self._paths = paths
        self._content = {p: "" for p in paths}
        self._pos = {p: 0 for p in paths}

    def refresh(self):
        for p in self._paths:
            with open(p, errors="replace") as f:
                f.seek(self._pos[p])
                new = f.read()
                self._pos[p] = f.tell()
            self._content[p] += new
        return self._content


def run_trial(endpoint, model, trial_idx, prompt_words, filler_count, filler_words, log_reader):
    print(f"\n--- Trial {trial_idx}: prompt={prompt_words}w, {filler_count} filler x {filler_words}w ---")
    prompt = make_prompt(random.randint(0, 2**31), prompt_words)

    print("  cold send (baseline, exercises the save/store path that crashed before)...")
    cold = send(endpoint, model, prompt)
    baseline_text = cold["choices"][0]["text"]
    baseline_tokens = cold.get("usage", {}).get("completion_tokens")
    print(f"    OK, no crash. completion_tokens={baseline_tokens}")

    print(f"  sending {filler_count} distinct filler prompts (unrelated, to create memory pressure)...")
    for _ in range(filler_count):
        send(endpoint, model, make_prompt(random.randint(0, 2**31), filler_words), max_tokens=1)

    print("  replay (same prompt, same deterministic params)...")
    replay = send(endpoint, model, prompt)
    replay_text = replay["choices"][0]["text"]
    replay_id = (replay.get("id") or "").replace("cmpl-", "")

    log_contents = log_reader.refresh()
    hit_info = lmcache_hit_tokens_for(replay_id, log_contents)

    if hit_info:
        engine_computed, lmcache_hit, total, worker = hit_info
        print(f"    LMCache log: engine_computed={engine_computed} lmcache_hit={lmcache_hit} "
              f"total={total} (worker={os.path.basename(worker)})")
    else:
        print("    WARNING: no matching 'Reqid: ... LMCache hit tokens' log line found "
              "(request may be shorter than chunk_size, served entirely by vLLM's own "
              "native GPU cache, or -- if this happens on every trial -- the completion-id "
              "<-> Reqid correlation this check depends on may have broken; see "
              "lmcache_hit_tokens_for()'s docstring)")

    correct = (replay_text == baseline_text)
    print(f"  token-for-token match: {'PASS' if correct else 'FAIL'}")
    if not correct:
        print(f"    baseline: {baseline_text[:200]!r}")
        print(f"    replay:   {replay_text[:200]!r}")

    return {
        "correct": correct,
        "hit_info": hit_info,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--logs", nargs="+", default=None)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--prompt-words", type=int, default=600)
    ap.add_argument("--filler-count", type=int, default=15)
    ap.add_argument("--filler-words", type=int, default=1500)
    ap.add_argument("--max-mismatches", type=int, default=2,
                     help="Tolerance for this stack's established LMCache-independent "
                          "floating-point noise floor (docs/local/LMCACHE_INTEGRATION_PLAN.md "
                          "§6: a same-N zero-LMCache control measured 0-1 mismatches per "
                          "10-15 trials). More than this many is treated as a real failure, "
                          "not noise. Default is 2 -- generous relative to the measured "
                          "baseline, not zero-tolerance.")
    args = ap.parse_args()

    cluster_env = load_env(os.path.join(REPO_ROOT, "config", "cluster.env"))
    model_env = load_env(os.path.join(REPO_ROOT, "config", "model.env"))

    endpoint = args.endpoint or f"http://{cluster_env['CONTROL_PLANE_IP']}:{cluster_env['VLLM_PORT']}/v1/completions"
    model = args.model or model_env["SERVED_MODEL_NAME"]
    log_paths = args.logs or discover_logs(args.n_workers)

    print(f"Endpoint: {endpoint}")
    print(f"Model:    {model}")
    print(f"Logs:     {log_paths}")

    try:
        with urllib.request.urlopen(endpoint.rsplit("/v1/", 1)[0] + "/v1/models", timeout=10) as resp:
            resp.read()
    except Exception as e:
        print(f"FAIL: endpoint not reachable: {e}")
        sys.exit(2)

    log_reader = _LogTailReader(log_paths)
    results = []
    for i in range(args.trials):
        results.append(run_trial(endpoint, model, i, args.prompt_words,
                                  args.filler_count, args.filler_words, log_reader))

    print("\n=== Summary ===")
    n_correct = sum(1 for r in results if r["correct"])
    n_mismatches = len(results) - n_correct
    n_lmcache_confirmed = sum(1 for r in results if r["hit_info"] and r["hit_info"][1] > 0)
    print(f"  token-for-token correct: {n_correct}/{len(results)}")
    print(f"  trials with a confirmed non-zero LMCache hit: {n_lmcache_confirmed}/{len(results)}")

    if n_mismatches > args.max_mismatches:
        print(f"  FAIL: {n_mismatches} mismatches exceeds the {args.max_mismatches}-trial "
              "tolerance for this stack's established floating-point noise floor -- treat as "
              "possible silent KV corruption, not noise.")
        sys.exit(1)
    if n_lmcache_confirmed == 0:
        print("  WARNING: no trial showed a confirmed LMCache hit (all correct, but the "
              "LMCache CPU tier itself may never have actually been exercised as the "
              "source of a replay -- consider more filler_count/prompt_words).")
    print(f"  PASS ({n_mismatches}/{len(results)} mismatches, within the "
          f"{args.max_mismatches}-trial noise-floor tolerance)")
    sys.exit(0)


if __name__ == "__main__":
    main()
