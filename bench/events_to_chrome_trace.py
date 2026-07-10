#!/usr/bin/env python3
"""Convert extract_request_events.py's per-request JSON into Chrome Trace
Event Format (viewable in chrome://tracing or https://ui.perfetto.dev/, and
mergeable with the RDU decode worker's coe_api trace via merge_traces.py).

Same phases already parsed for the matplotlib plot (queue / prefill compute
/ KV transfer / decode) become "X" (complete) events -- one per request per
phase. Each request gets its own thread lane (`tid`) inside its worker's
process lane (`pid`), so concurrent requests (conc > 1) never need
stack-based B/E nesting or visually overlap on the same lane.

Timestamps are converted from extract_request_events.py's unix-epoch-seconds
to epoch MICROSECONDS -- the same absolute reference frame coe_api's own
Profiler uses (CLOCK_REALTIME), so a GPU-side trace and an RDU-side trace
line up correctly once merged, with no separate calibration step, as long as
the cluster's nodes are NTP-synced.

Usage:
  python3 events_to_chrome_trace.py events.json -o gpu_trace.json
"""
import argparse
import json
import sys

# (start_field, end_field, phase_name) -- same fields extract_request_events.py
# already populates; a phase is only emitted if both timestamps are present.
# decode_queue (t_kv_ready -> t_xfer_start) covers the gap between prefill
# finishing and the decode worker actually starting to pull KV -- confirmed
# earlier this session to be the DOMINANT phase in a decode-capacity-bound
# run (16-64s, dwarfing prefill/transfer/decode combined), driven by vLLM's
# own scheduler gating transfer-initiation on a free running slot
# (max_num_seqs), not by network/transfer speed. Previously this showed up
# only as unlabeled blank space between the prefill and kv_transfer bars.
PHASES = [
    ("t_received", "t_started", "queue"),
    ("t_started", "t_kv_ready", "prefill"),
    ("t_kv_ready", "t_xfer_start", "decode_queue"),
    ("t_xfer_start", "t_xfer_end", "kv_transfer"),
    ("t_xfer_end", "t_completed_decode", "decode"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("events", help="JSON from extract_request_events.py")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    with open(args.events) as f:
        records = json.load(f)

    worker_pid = {}
    next_pid = 1
    next_tid_for_pid = {}
    trace_events = []

    for rec in records:
        worker = rec.get("worker")
        if worker is None:
            continue
        if worker not in worker_pid:
            pid = next_pid
            next_pid += 1
            worker_pid[worker] = pid
            next_tid_for_pid[pid] = 1
            trace_events.append(
                {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": worker}}
            )
        pid = worker_pid[worker]
        tid = next_tid_for_pid[pid]
        next_tid_for_pid[pid] += 1

        req_id = rec.get("request_id", "?")
        trace_events.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": f"req {req_id[:8]}"},
            }
        )

        for start_key, end_key, name in PHASES:
            t0 = rec.get(start_key)
            t1 = rec.get(end_key)
            if t0 is None or t1 is None:
                continue
            trace_events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": t0 * 1e6,
                    "dur": max((t1 - t0) * 1e6, 0.0),
                    "name": name,
                    "args": {"request_id": req_id},
                }
            )

    out = {"traceEvents": trace_events, "displayTimeUnit": "ms"}
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(
        f"Wrote {len(trace_events)} trace events ({len(worker_pid)} workers) -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
