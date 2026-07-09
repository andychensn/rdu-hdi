#!/usr/bin/env python3
"""E2E test: prove Dynamo's KV-cache-aware + load-aware prefill routing
(--router-mode kv, set up in docker/control-plane/entrypoint.sh and
launch/gpu_prefill.sh) is actually active against a live stack, not just
configured.

Assumes control-plane + >=2 co-located GPU prefill workers + decode are
ALREADY running (see launch/control_plane.sh, launch/gpu_prefill.sh,
launch/rdu_decode.sh) -- same convention as test/e2e_rdu_decode.sh. Sends
real requests against the live endpoint and reads each prefill worker's own
log to determine which worker actually handled each request.

Usage:
  python3 test/e2e_kv_routing.py
  python3 test/e2e_kv_routing.py --logs logs/A_gpu_prefill_0.log logs/B_gpu_prefill_1.log
  python3 test/e2e_kv_routing.py --endpoint http://otherhost:18000 --model MiniMax-M2.7

Two checks:
  1. Cache-affinity: N independent trials, each sending a FRESH
     never-before-seen prefix once (cold) then immediately repeating it
     once, checking whether the repeat lands on the same worker as the cold
     request. Plain round-robin assigns purely by arrival position, so for
     this cold-then-repeat pairing it is fully deterministic: cold always
     lands at an odd position, repeat always at the next even position --
     different workers, every single trial. Round-robin's stickiness rate
     for this design is therefore exactly 0%, always -- no aliasing
     ambiguity. Real cache-aware routing should show a sticky rate well
     above that floor (though not necessarily 100%, since
     --router-temperature intentionally adds sampling noise so it doesn't
     hard-pin traffic).
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
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()


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


def check_cache_affinity(endpoint, model, log_paths, n_trials=10, min_sticky_rate=0.3):
    print("\n=== Check 1: cache-affinity (many independent cold+repeat trials) ===")
    # Design: N independent trials, each with a FRESH never-before-seen
    # prefix sent once (cold) then immediately repeated once. Checks whether
    # the repeat lands on the same worker as the cold request.
    #
    # Why this shape and not "repeat the same prefix K times in a row"
    # (tried first, see git history): repeating in a row creates CORRELATED
    # samples -- if an early repeat happens to miss (temperature-driven
    # sampling, not a bug -- confirmed live by inspecting prefill_elapsed_ms:
    # a genuine cold miss on the "wrong" worker), that miss warms the wrong
    # worker too, and every subsequent repeat in that same run can then keep
    # landing there "by chance" since both workers are now equally cached.
    # One bad early sample dominates the whole trial. Fresh-prefix-per-trial
    # avoids this entirely -- every trial is independent.
    #
    # Why this is also unambiguous against a round-robin regression (unlike
    # the ABAB design tried first): each trial is exactly 2 requests, cold
    # then repeat. Plain round-robin assigns purely by arrival position --
    # cold always lands at an odd position (-> worker0), repeat always at
    # the next even position (-> worker1), for EVERY trial, deterministically.
    # So round-robin's stickiness rate for this design is exactly 0%, always
    # -- there's no aliasing risk like the ABAB design had. Any real
    # stickiness above a small noise floor is a genuine signal.
    sticky = 0
    trials_completed = 0
    for i in range(n_trials):
        prefix = make_prefix(random.randint(0, 2**31), 13000)
        before = {p: len(parse_log_events(p)) for p in log_paths}
        for _ in range(2):  # cold, then immediate repeat
            suffix = f" uniquepart{random.randint(0, 10**9)} " * 50
            send(endpoint, model, prefix + suffix)
        time.sleep(0.3)

        new_events = []
        for p in log_paths:
            all_events = parse_log_events(p)
            for e in all_events[before[p]:]:
                new_events.append((p, *e))
        new_events.sort(key=lambda e: e[1])
        if len(new_events) < 2:
            print(f"  trial {i+1}: WARNING only found {len(new_events)}/2 request events, skipping")
            continue
        cold_worker, repeat_worker = new_events[-2][0], new_events[-1][0]
        trials_completed += 1
        is_sticky = cold_worker == repeat_worker
        sticky += int(is_sticky)
        print(f"  trial {i+1}: cold->{os.path.basename(cold_worker)}, "
              f"repeat->{os.path.basename(repeat_worker)} {'(sticky)' if is_sticky else '(exception)'}")

    if trials_completed == 0:
        print("  FAIL: no trials completed -- could not observe any requests in the worker logs.")
        return False

    rate = sticky / trials_completed
    print(f"  sticky rate: {sticky}/{trials_completed} ({rate:.0%}); "
          f"round-robin's deterministic baseline for this design is 0%")

    if rate < min_sticky_rate:
        print(f"  FAIL: sticky rate {rate:.0%} is below the {min_sticky_rate:.0%} threshold -- "
              "routing doesn't show meaningful cache preference (could be round-robin, could be "
              "load/temperature fully dominating cache credit).")
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

    ok1 = check_cache_affinity(endpoint, model, log_paths)
    ok2 = check_load_balance(endpoint, model, log_paths)

    print("\n=== Summary ===")
    print(f"  cache-affinity check: {'PASS' if ok1 else 'FAIL'}")
    print(f"  load-balance check:   {'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
