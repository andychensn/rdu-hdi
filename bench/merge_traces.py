#!/usr/bin/env python3
"""Merge N Chrome Trace Event JSON files (e.g. one per GPU prefill worker
from events_to_chrome_trace.py, plus the RDU decode worker's own coe_api
save_profile() dump) into ONE combined trace, viewable in chrome://tracing
or https://ui.perfetto.dev/ with every physical worker as its own labeled
process lane.

Deliberately NOT SambaNova's own `snprof -cj/--combine_json` -- that tool
just concatenates traceEvents arrays with no pid remapping at all, and
coe_api's own `pid` field is the raw OS getpid() (see
docs/local/CLUSTER_PROFILING_DESIGN.md section 1a) -- two unrelated
processes on two different physical hosts can trivially collide on the same
numeric pid, which would silently merge their events onto the same visual
track. This script remaps every source file's pid space to fresh,
globally-unique pids first, and labels each with the source's own name via
a "process_name" metadata event.

Relies on every source using the same absolute time reference (epoch
microseconds) -- coe_api's Profiler uses CLOCK_REALTIME, and
events_to_chrome_trace.py converts extract_request_events.py's epoch
seconds the same way, so this holds as long as cluster nodes are NTP-synced.
No calibration is attempted here; sanity-check cross-referenced events (e.g.
a request's transfer-end vs. its RDU-side receive) after merging if the
result looks implausible.

Usage:
  python3 merge_traces.py worker0=gpu_worker0_trace.json \\
      worker1=gpu_worker1_trace.json \\
      rdu_decode0=rdu_decode_trace.json \\
      -o combined_trace.json
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sources", nargs="+", help="label=path.json pairs")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    combined = []
    next_pid = 1

    for spec in args.sources:
        if "=" not in spec:
            sys.exit(f"ERROR: expected label=path, got {spec}")
        label, path = spec.split("=", 1)
        with open(path) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        if not events:
            print(f"WARNING: {label} ({path}) has no traceEvents -- skipping", file=sys.stderr)
            continue

        old_pids = sorted({e["pid"] for e in events if "pid" in e})
        pid_map = {}
        for old_pid in old_pids:
            pid_map[old_pid] = next_pid
            next_pid += 1

        for e in events:
            if "pid" in e:
                e["pid"] = pid_map[e["pid"]]
            combined.append(e)

        for old_pid, new_pid in pid_map.items():
            name = label if len(pid_map) == 1 else f"{label} (pid {old_pid})"
            combined.append(
                {"ph": "M", "pid": new_pid, "name": "process_name", "args": {"name": name}}
            )
        print(
            f"{label}: {len(events)} events, {len(pid_map)} pid(s) -> {sorted(pid_map.values())}",
            file=sys.stderr,
        )

    if not combined:
        sys.exit("ERROR: no valid trace events found in any source")

    out = {"traceEvents": combined, "displayTimeUnit": "ms"}
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(combined)} combined trace events -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
