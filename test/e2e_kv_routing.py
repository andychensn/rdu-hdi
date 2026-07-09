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
  2. Load-balance: sends N requests SEQUENTIALLY, each pairing one FIXED
     shared prefix with its own unique random suffix -- a realistic traffic
     shape (shared system prompt + different user turns). Since the suffix
     is always new work regardless of worker, the shared prefix can only
     ever cover about half of a request's cost, capping the cache-holding
     worker's advantage -- enough for the router's temperature-based
     selection to naturally spread traffic across workers over many
     requests, without needing concurrency or an engineered busy worker.

Exit code: 0 = both checks pass, 1 = a check failed, 2 = setup/discovery error.
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


def check_load_balance(endpoint, model, log_paths, n_requests=60, prefix_words=1350,
                        suffix_words=1350, max_worker_share=0.85):
    print("\n=== Check 2: load-balance (fixed prefix + random suffix, sequential) ===")
    # Design: one FIXED shared prefix (~4k tokens) reused across every
    # request, each paired with its own unique random suffix (~4k tokens),
    # sent fully sequentially -- a realistic traffic shape (shared system
    # prompt + different user turns).
    #
    # This deliberately does NOT try to engineer concurrent load or an
    # artificially busy worker. An earlier, much more elaborate design tried
    # exactly that (fire a large background request to occupy one worker,
    # then send concurrent "probe" requests) and it did work, but only after
    # working around two confounds: the background job's own worker
    # placement is itself close to a coin flip unless you wait for its
    # KV-store event to actually propagate to the router first, and large
    # probe requests risk tripping the router's hard busy-instance reject
    # (503) if aimed at an already-busy worker.
    #
    # Because the random suffix is always new work regardless of which
    # worker handles it, the shared prefix can only ever cover about half of
    # a given request's cost -- capping how large a cost advantage the
    # cache-holding worker can have. With router_temperature=0.4 (softmax
    # sampling, not deterministic argmin), that bounded advantage is enough
    # on its own for traffic to spread across every worker over many
    # requests, with both workers showing real cache hits once each has
    # served the prefix at least once -- no concurrency needed at all.
    #
    # n_requests=60, not 20: at n=20 this was genuinely flaky, not just
    # slow to converge -- 10 live runs at n=20 ranged from a clean 50/50
    # split down to 2 runs where ALL 20 requests landed on one worker
    # (0/20 and 5/20 on the other), consistent with an early near-tie
    # sometimes cascading into a self-reinforcing streak for the rest of a
    # short run, rather than a smoothly-converging average. 10 live runs at
    # n=60 ranged 52-77% max-worker-share -- comfortably clear of
    # max_worker_share below, with real margin, and no full lock-in in any
    # of the 10. If this starts flaking again (e.g. after a config change
    # that shifts router_temperature or the credit weights), raising
    # n_requests further is the first thing to try before assuming a real
    # regression.
    print(f"  sending {n_requests} sequential requests sharing one fixed ~{prefix_words}-word "
          f"prefix, each with its own random ~{suffix_words}-word suffix...")
    fixed_prefix = make_prefix(random.randint(0, 2**31), prefix_words)

    req_ids = []
    hits = 0
    for _ in range(n_requests):
        suffix = make_prefix(random.randint(0, 2**31), suffix_words)
        resp = send(endpoint, model, fixed_prefix + " " + suffix)
        hits += int(bool(cached_tokens_of(resp)))
        req_ids.append((resp.get("id") or "").replace("cmpl-", ""))

    print("  batch complete. correlating each request to a worker via its response id "
          "in the worker logs...")
    log_contents = {}
    for path in log_paths:
        with open(path, errors="replace") as f:
            log_contents[path] = f.read()

    counts = defaultdict(int)
    unmatched = 0
    for req_id in req_ids:
        worker = next((p for p, content in log_contents.items() if req_id and req_id in content), None)
        if worker is None:
            unmatched += 1
        else:
            counts[worker] += 1

    if unmatched:
        print(f"  WARNING: {unmatched}/{n_requests} response(s) could not be matched to a "
              "worker log by request id (log may be stale/rotated mid-run).")

    total = sum(counts.values())
    if total == 0:
        print("  FAIL: no requests could be matched to any worker log.")
        return False

    for path in log_paths:
        c = counts.get(path, 0)
        print(f"  {os.path.basename(path)}: {c}/{total} ({c/total:.0%})")
    print(f"  cache-hit rate across the run: {hits}/{n_requests} (both workers should show "
          "hits once each has served this prefix at least once)")

    max_share = max(counts.values()) / total
    print(f"  max single-worker share: {max_share:.0%} (threshold: {max_worker_share:.0%})")
    if max_share > max_worker_share:
        print("  FAIL: nearly all traffic stuck on one worker -- load isn't being distributed "
              "across workers.")
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
