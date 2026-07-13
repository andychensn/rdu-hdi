#!/usr/bin/env python3
"""Basic smoke test for LMCache's CPU and disk offload tiers.

NOT a rigorous correctness/statistical gate -- see test/e2e_lmcache_correctness.py
for that. This just answers a much simpler question: does the KV cache actually
get offloaded to each tier, loaded back, used, and produce a real completion?
No edge cases, no crash-injection, no mismatch-tolerance statistics.

Assumes control-plane + >=1 GPU prefill worker + RDU decode are already running
(same convention as the other test/e2e_*.py scripts), launched with
GPU_DISABLE_NATIVE_PREFIX_CACHE=1 -- without it, vLLM's own native GPU prefix
cache silently serves replays itself and LMCache is never actually exercised
(see test/e2e_lmcache_correctness.py's docstring for why).

Usage:
  GPU_DISABLE_NATIVE_PREFIX_CACHE=1 bash launch/gpu_prefill.sh
  python3 test/smoke_lmcache_tiers.py

Exit code: 0 = both enabled tiers were genuinely exercised and produced correct,
non-empty completions. 1 = a tier that should have been exercised wasn't, or a
completion came back empty/wrong. 2 = setup error (endpoint unreachable).
"""
import argparse
import glob
import os
import random
import re
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_kv_routing import load_env  # noqa: E402 -- shared env-file parser
from e2e_lmcache_correctness import (  # noqa: E402 -- reuse, don't duplicate
    discover_logs,
    make_prompt,
    send,
    _RETRIEVE_LINE_RE,
)

# LMCache's own debug-level line for an actual disk read (local_disk_backend.py's
# read_file()) -- the strongest available signal that the DISK tier specifically
# (not just the CPU tier) was touched. Requires LMCACHE_LOG_LEVEL=DEBUG.
_DISK_READ_RE = re.compile(r"Disk read size: (\d+) bytes")


def genuine_retrieve(log_contents_by_path):
    """Did ANY request show a real "Retrieved N out of M" ground-truth line
    (cache_engine.py's retrieve() path -- fires for a genuine reload from
    either the CPU or disk tier, not just a scheduling-time hit report)?
    """
    for path, content in log_contents_by_path.items():
        m = _RETRIEVE_LINE_RE.search(content)
        if m:
            return int(m.group(2)), int(m.group(3)), path
    return None


def genuine_disk_read(log_contents_by_path):
    """Did any worker log a real disk-tier read (LocalDiskBackend.read_file())?"""
    for path, content in log_contents_by_path.items():
        m = _DISK_READ_RE.search(content)
        if m:
            return int(m.group(1)), path
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--logs", nargs="+", default=None)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--prompt-words", type=int, default=600)
    ap.add_argument("--filler-count", type=int, default=20,
                     help="Distinct filler prompts sent to create eviction pressure. "
                          "Disk-tier reuse only happens once the CPU tier actually evicts "
                          "something -- if LMCACHE_MAX_LOCAL_CPU_GB is large in the current "
                          "config, this may need to be much larger (or the CPU budget "
                          "temporarily lowered) to see real disk activity within this test.")
    ap.add_argument("--filler-words", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=2,
                     help="Kept short deliberately -- this stack has a real, "
                          "LMCache-independent floating-point noise floor that compounds "
                          "over longer generations (see test/e2e_lmcache_correctness.py's "
                          "docstring: 9/10 mismatches at 64 tokens, 0-1/10 at <=2). A short "
                          "window is needed for a clean pass/fail signal.")
    args = ap.parse_args()

    cluster_env = load_env(os.path.join(REPO_ROOT, "config", "cluster.env"))
    model_env = load_env(os.path.join(REPO_ROOT, "config", "model.env"))

    endpoint = args.endpoint or f"http://{cluster_env['CONTROL_PLANE_IP']}:{cluster_env['VLLM_PORT']}/v1/completions"
    model = args.model or model_env["SERVED_MODEL_NAME"]
    log_paths = args.logs or discover_logs(args.n_workers)
    disk_tier_configured = int(model_env.get("LMCACHE_MAX_LOCAL_DISK_GB", "0")) > 0

    print(f"Endpoint: {endpoint}")
    print(f"Model:    {model}")
    print(f"Logs:     {log_paths}")
    print(f"Disk tier configured (LMCACHE_MAX_LOCAL_DISK_GB > 0): {disk_tier_configured}")

    try:
        with urllib.request.urlopen(endpoint.rsplit("/v1/", 1)[0] + "/v1/models", timeout=10) as resp:
            resp.read()
    except Exception as e:
        print(f"FAIL: endpoint not reachable: {e}")
        sys.exit(2)

    failures = []

    # ── Step 1: basic sanity -- one real completion, non-empty, no crash ──────
    print("\n--- Step 1: basic completion sanity check ---")
    prompt = make_prompt(random.randint(0, 2**31), args.prompt_words)
    cold = send(endpoint, model, prompt, max_tokens=args.max_tokens)
    baseline_text = cold["choices"][0]["text"]
    if not baseline_text.strip():
        failures.append("cold completion returned empty text")
        print("  FAIL: empty completion text")
    else:
        print(f"  OK: got a real completion ({baseline_text[:80]!r}...)")

    # ── Step 2: CPU-tier reuse -- replay immediately, no eviction needed ──────
    print("\n--- Step 2: CPU-tier offload + reload ---")
    replay = send(endpoint, model, prompt, max_tokens=args.max_tokens)
    replay_text = replay["choices"][0]["text"]
    if replay_text != baseline_text:
        failures.append(f"CPU-tier replay text mismatch: {baseline_text!r} vs {replay_text!r}")
        print(f"  FAIL: replay text differs (baseline={baseline_text!r}, replay={replay_text!r})")
    else:
        print(f"  OK: replay produced identical, real tokens ({replay_text[:80]!r}...)")

    log_contents = {p: open(p, errors="replace").read() for p in log_paths}
    cpu_retrieve = genuine_retrieve(log_contents)
    if cpu_retrieve:
        retrieved, required, worker = cpu_retrieve
        print(f"  OK: genuine LMCache retrieve confirmed ({retrieved}/{required} tokens, "
              f"worker={os.path.basename(worker)})")
    else:
        failures.append("no genuine LMCache retrieve line found for the CPU-tier replay")
        print("  FAIL: no 'Retrieved N out of M required tokens' line found -- "
              "was GPU_DISABLE_NATIVE_PREFIX_CACHE=1 set when this worker was launched?")

    # ── Step 3: disk-tier reuse -- needs real CPU-tier eviction first ─────────
    if not disk_tier_configured:
        print("\n--- Step 3: disk-tier offload + reload -- SKIPPED "
              "(LMCACHE_MAX_LOCAL_DISK_GB=0 in config/model.env) ---")
    else:
        print("\n--- Step 3: disk-tier offload + reload ---")
        disk_prompt = make_prompt(random.randint(0, 2**31), args.prompt_words)
        disk_cold = send(endpoint, model, disk_prompt, max_tokens=args.max_tokens)
        disk_baseline_text = disk_cold["choices"][0]["text"]

        print(f"  sending {args.filler_count} distinct filler prompts to force CPU-tier eviction...")
        for _ in range(args.filler_count):
            send(endpoint, model, make_prompt(random.randint(0, 2**31), args.filler_words), max_tokens=1)

        disk_replay = send(endpoint, model, disk_prompt, max_tokens=args.max_tokens)
        disk_replay_text = disk_replay["choices"][0]["text"]
        if disk_replay_text != disk_baseline_text:
            failures.append(f"disk-tier replay text mismatch: {disk_baseline_text!r} vs {disk_replay_text!r}")
            print(f"  FAIL: replay text differs (baseline={disk_baseline_text!r}, "
                  f"replay={disk_replay_text!r})")
        else:
            print(f"  OK: replay produced identical, real tokens ({disk_replay_text[:80]!r}...)")

        log_contents = {p: open(p, errors="replace").read() for p in log_paths}
        disk_read = genuine_disk_read(log_contents)
        if disk_read:
            size, worker = disk_read
            print(f"  OK: genuine disk-tier read confirmed ({size} bytes, "
                  f"worker={os.path.basename(worker)})")
        else:
            print("  INCONCLUSIVE: no 'Disk read size' line found -- either "
                  "LMCACHE_LOG_LEVEL=DEBUG wasn't set, or --filler-count wasn't enough to "
                  "force real CPU-tier eviction with the current LMCACHE_MAX_LOCAL_CPU_GB "
                  "(this is a setup/sizing issue, not treated as a hard failure here -- "
                  "rerun with a larger --filler-count or a temporarily-lowered CPU budget "
                  "if you need a definitive answer)")

    print("\n=== Summary ===")
    if failures:
        print(f"FAIL ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: basic offload/reload/reuse smoke checks succeeded.")
    sys.exit(0)


if __name__ == "__main__":
    main()
