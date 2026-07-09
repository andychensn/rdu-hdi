#!/usr/bin/env python3
"""E2E test: prove Dynamo's KV-cache-aware + load-aware prefill routing
(--router-mode kv, set up in docker/control-plane/entrypoint.sh and
launch/gpu_prefill.sh) is actually active against a live stack, not just
configured.

Assumes control-plane + >=2 co-located GPU prefill workers + decode are
ALREADY running (see launch/control_plane.sh, launch/gpu_prefill.sh,
launch/rdu_decode.sh) -- same convention as test/e2e_rdu_decode.sh. Sends
real requests against the live endpoint.

Usage:
  python3 test/e2e_kv_routing.py
  python3 test/e2e_kv_routing.py --logs logs/A_gpu_prefill_0.log logs/B_gpu_prefill_1.log
  python3 test/e2e_kv_routing.py --endpoint http://otherhost:18000 --model MiniMax-M2.7

Two checks:
  1. Cache-affinity: sends N unique prompts (cold), waits for the entire
     batch to finish, shuffles the order, then resends all N as repeats --
     reading the cache-hit signal straight from each repeat's own API
     response (usage.prompt_tokens_details.cached_tokens, sourced from
     vLLM's own request_output.num_cached_tokens -- see send()'s docstring
     for how this was confirmed to actually work in this deployment). The
     right null hypothesis is RANDOM choice among the N_WORKERS prefill
     workers, not round-robin: if the router ignored cache state and picked
     a worker uniformly at random for each repeat, it would land on the
     cache-holding worker ~1/N_WORKERS of the time by chance alone -- so a
     raw hit rate above 0% proves nothing; only a rate significantly above
     1/N_WORKERS does. Runs a one-sided exact binomial test against that
     baseline rather than an arbitrary percentage cutoff.
  2. Load-balance: sends N requests at concurrency>1 sharing one prefix,
     checks traffic isn't pinned onto a single worker despite established
     cache affinity (that would indicate the load term isn't working).

Exit code: 0 = both checks pass, 1 = a check failed, 2 = setup/discovery error.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(*paths):
    env = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or "=" not in line or line.startswith("("):
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def discover_logs(n_workers):
    """Find the most recent gpu_prefill_*.log group (same launch timestamp)."""
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
        raise SystemExit("No logs/*_gpu_prefill_*.log found -- is a GPU prefill worker running? "
                          "(launch/gpu_prefill.sh writes these)")
    latest_ts = max(groups)
    group = groups[latest_ts]
    if len(group) < n_workers:
        raise SystemExit(f"Most recent launch ({latest_ts}) only has {len(group)} worker log(s), "
                          f"need >= {n_workers}. Pass --logs explicitly if this is stale.")
    return [group[i] for i in sorted(group)]


def make_prefix(seed, n_words):
    rng = random.Random(seed)
    return " ".join(f"tok{rng.randint(0, 999999)}" for _ in range(n_words))


def send(endpoint, model, prompt, max_tokens=5):
    """Returns the parsed JSON response (not just fire-and-forget) -- Dynamo's
    vLLM handler (components/src/dynamo/vllm/handlers.py) puts a real,
    ground-truth cache-hit signal in the response itself:
    usage.prompt_tokens_details.cached_tokens, sourced directly from vLLM's
    own request_output.num_cached_tokens. Confirmed live: a short (11-token)
    repeat showed no such field (block_size=64 means anything under one
    block can never register a hit, so absence there is expected, not
    evidence the mechanism is broken), but a 600-token prompt repeated
    showed cached_tokens=576. Also confirmed from source
    (dynamo/vllm/handlers.py) that the field is only included when
    truthy -- Python's `if 0:` is False, so a genuine miss (0 cached
    tokens) and an unsupported/absent attribute are indistinguishable by
    absence alone. Use long-enough prompts (multiple blocks) and don't
    read anything into absence beyond "not a clear hit."
    """
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def cached_tokens_of(response):
    details = response.get("usage", {}).get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or 0


def _comb(n, k):
    """n-choose-k, no math.comb dependency (needs Python 3.8+; this repo's
    login-node default `python3` is 3.6)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def binomial_sf(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p), computed exactly (no scipy dependency)."""
    return sum(_comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def parse_log_events(log_path):
    """Return [(unix_ts, prompt_tokens, elapsed_ms), ...] sorted by time."""
    # kv_ready lines are plain vLLM engine log lines, e.g.:
    #   (EngineCore_DP0 pid=954) INFO 07-09 00:04:46 [nixl_connector.py:1004]
    #   [kv_ready] req=b7e61a3a blocks=615 prompt_tokens=39342 prefill_elapsed_ms=47
    # -- MM-DD HH:MM:SS, no year, no ANSI codes. Good enough to order events
    # within a single short-lived test run (this script's own scope).
    ready_re = re.compile(
        r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[kv_ready\] req=(\S+) blocks=\d+ "
        r"prompt_tokens=(\d+) prefill_elapsed_ms=(\d+)"
    )
    events = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = ready_re.search(line)
            if m:
                ts, _req, ptok, ms = m.groups()
                events.append((ts, int(ptok), int(ms)))
    return events


def check_cache_affinity(endpoint, model, n_workers, n_prompts=20, prompt_words=2650,
                          alpha=0.05):
    print("\n=== Check 1: cache-affinity (batch cold, then shuffled repeat batch) ===")
    # Design: send N unique prompts (cold) fully SEQUENTIALLY, wait for the
    # entire batch to finish, shuffle the order, then resend all N as
    # repeats -- also fully sequentially. Read cached_tokens straight from
    # each repeat's own API response (see send()/cached_tokens_of()).
    #
    # Both phases are sequential, not concurrent, on purpose: an earlier
    # version sent the repeat phase 8-at-a-time and measured a hit rate
    # BELOW the random-chance baseline, which has no plausible explanation
    # as a routing bug (a broken router would look random, not
    # anti-correlated) but is exactly what self-induced load competition
    # would produce -- with several *other* repeats concurrently in flight,
    # the "correct" (cache-holding) worker's load term can legitimately
    # spike at the exact moment a given repeat is routed, causing the
    # router to correctly route AWAY from it for load reasons. That's this
    # check accidentally exercising the load-balance mechanism (tested
    # separately, on purpose, in check_load_balance) at the same time it's
    # trying to isolate cache-affinity alone. Sequential sending removes
    # that confound entirely -- only one request is ever in flight, so
    # there's no artificial load signal from this test's own traffic.
    #
    # The right null hypothesis is RANDOM choice among n_workers, not
    # round-robin: if the router ignored cache state and picked a worker
    # uniformly at random for the repeat, it would land on the cache-holding
    # worker ~1/n_workers of the time by chance alone. A raw hit rate above
    # 0% proves nothing; only a rate meaningfully above 1/n_workers does.
    # Using an exact one-sided binomial test against that baseline rather
    # than an arbitrary percentage threshold, since the raw threshold would
    # itself need justifying -- the binomial test says exactly how
    # surprising the observed count would be under pure random chance,
    # which is the actual question.
    prompts = [make_prefix(random.randint(0, 2**31), prompt_words) for _ in range(n_prompts)]

    print(f"  sending {n_prompts} unique cold prompts, sequentially...")
    cold_responses = [send(endpoint, model, p) for p in prompts]
    cold_tokens = [r.get("usage", {}).get("prompt_tokens", 0) for r in cold_responses]

    order = list(range(n_prompts))
    random.shuffle(order)
    print(f"  batch complete. Resending all {n_prompts} in shuffled order as repeats, "
          f"sequentially...")
    repeat_responses = [send(endpoint, model, prompts[i]) for i in order]

    hits = 0
    for i, resp in zip(order, repeat_responses):
        cached = cached_tokens_of(resp)
        # require the hit to cover a real majority of the prompt, not a
        # tiny/spurious overlap -- vLLM's own block-level caching means a
        # genuine same-worker repeat should cover nearly the whole prompt
        # (minus one trailing partial block), not a token or two.
        is_hit = cached > 0.5 * cold_tokens[i]
        hits += int(is_hit)

    baseline = 1 / n_workers
    print(f"  hits: {hits}/{n_prompts} ({hits/n_prompts:.0%}); random-choice baseline for "
          f"{n_workers} workers is {baseline:.0%}; perfect stickiness would be 100%")

    p_value = binomial_sf(n_prompts, hits, baseline)
    print(f"  one-sided binomial test vs. random chance: p={p_value:.4f} (need < {alpha})")

    if p_value >= alpha:
        print(f"  FAIL: hit rate is not significantly above the {baseline:.0%} random-chance "
              "baseline -- routing doesn't show meaningful cache preference.")
        return False
    print("  PASS")
    return True


def check_load_balance(endpoint, model, log_paths, n_requests=20, concurrency=4, max_worker_share=0.8):
    print("\n=== Check 2: load-balance (shared prefix, concurrent burst) ===")
    prefix = make_prefix(random.randint(0, 2**31), 13000)  # fresh every run, see check_cache_affinity's note
    before = {p: len(parse_log_events(p)) for p in log_paths}

    def one_request(_i):
        suffix = f" uniquepart{random.randint(0, 10**9)} " * 50
        send(endpoint, model, prefix + suffix)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one_request, range(n_requests)))

    time.sleep(1)
    counts = {}
    for p in log_paths:
        counts[p] = len(parse_log_events(p)) - before[p]

    total = sum(counts.values())
    if total < n_requests:
        print(f"  FAIL: expected {n_requests} new completions, only found {total}.")
        return False

    for p, c in counts.items():
        print(f"  {os.path.basename(p)}: {c}/{total} ({c/total:.0%})")

    max_share = max(counts.values()) / total
    print(f"  max single-worker share: {max_share:.0%} (threshold: {max_worker_share:.0%})")
    if max_share > max_worker_share:
        print("  FAIL: one worker took more than the allowed share -- load term may not be working "
              "(all cache-hit traffic pinned to one worker).")
        return False
    print("  PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None, help="e.g. http://10.10.0.156:18000 (default: from config/cluster.env)")
    ap.add_argument("--model", default=None, help="default: from config/model.env's SERVED_MODEL_NAME")
    ap.add_argument("--logs", nargs="+", default=None,
                    help="explicit prefill worker log paths (default: auto-discover the most recent launch)")
    ap.add_argument("--n-workers", type=int, default=2)
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

    ok1 = check_cache_affinity(endpoint, model, args.n_workers)
    ok2 = check_load_balance(endpoint, model, log_paths)

    print("\n=== Summary ===")
    print(f"  cache-affinity check: {'PASS' if ok1 else 'FAIL'}")
    print(f"  load-balance check:   {'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
