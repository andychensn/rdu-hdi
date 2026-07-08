#!/usr/bin/env python3
"""Parse prefill + decode worker log(s) into a full per-request timeline.

Extracts, per request_id (matched on the first 8 hex chars — every log format
here truncates the UUID to at least that length):

  From the prefill worker's log:
    - `request received` (dynamo ingress, UTC, sub-second) -- when the
      request reaches the GPU node and starts queueing
    - `prefill.req.started` (NIXL connector, UTC, second-granularity) --
      when the engine actually starts computing (queue_wait = started - received)
    - `kv_ready ... prefill_elapsed_ms=N` (NIXL connector) -- when prefill
      compute finishes and KV is ready to be pulled
    - `request completed` (dynamo ingress) -- prefill-side completion

  From the decode worker's log (--decode; requires VLLM_RDU_PLUGIN_TIME_PROFILE=1
  on the RDU decode container for the transfer timing line to exist at all):
    - `request received` (dynamo ingress, component="backend") -- when the
      request reaches the decode/RDU node
    - `[time] nixl.transfer elapsed_ms=N req=...` -- KV transfer duration;
      t_xfer_end = this log line's own timestamp, t_xfer_start = t_xfer_end - N/1000
    - `request completed` (dynamo ingress, component="backend") -- full
      end-to-end completion (after decode finishes generating)

Output: a JSON list of records, one per request, unix-epoch-UTC (float) timestamps:
  {worker, request_id,
   t_received, t_started, t_kv_ready, compute_ms, t_completed,
   t_received_decode, t_xfer_start, t_xfer_end, xfer_elapsed_ms, t_completed_decode}

Usage:
  python3 extract_request_events.py worker0=/path/to/gpu_prefill_0.log \\
      worker1=/path/to/gpu_prefill_1.log \\
      --decode /path/to/rdu_decode.log \\
      --start 1783486038.86 --end 1783486398.19 -o events.json

Note: the connector's non-ISO timestamps (`MM-DD HH:MM:SS`, no year) --
pass --year if analyzing logs from a year other than the default.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

RECEIVED_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}\.\d+)Z.*?request received "
    r"request_id=([0-9a-f-]+) "
)
COMPLETED_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}\.\d+)Z.*?request completed "
    r"request_id=([0-9a-f-]+) "
)
STARTED_RE = re.compile(
    r"(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[nixl_connector.*?"
    r"prefill\.req\.started req=(\S+?)-?\s+prompt_tokens=(\d+)"
)
KV_READY_RE = re.compile(
    r"(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[nixl_connector.*?"
    r"\[kv_ready\] req=(\S+) blocks=\d+ prompt_tokens=\d+ prefill_elapsed_ms=(\d+)"
)
XFER_RE = re.compile(
    r"(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[connector_override.*?"
    r"\[time\] nixl\.transfer elapsed_ms=([\d.]+) req=(\S+)"
)


def parse_iso(date_s, time_s):
    dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def parse_short(mmdd, hms, year):
    dt = datetime.strptime(f"{year}-{mmdd} {hms}", "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def parse_prefill_log(path, worker_label, year):
    received, completed, started, kv_ready = {}, {}, {}, {}
    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            m = RECEIVED_RE.search(line)
            if m:
                received[m.group(3)[:8]] = parse_iso(m.group(1), m.group(2))
                continue
            m = COMPLETED_RE.search(line)
            if m:
                completed[m.group(3)[:8]] = parse_iso(m.group(1), m.group(2))
                continue
            m = STARTED_RE.search(line)
            if m:
                started[m.group(3)[:8]] = parse_short(m.group(1), m.group(2), year)
                continue
            m = KV_READY_RE.search(line)
            if m:
                kv_ready[m.group(3)[:8]] = (
                    parse_short(m.group(1), m.group(2), year),
                    int(m.group(4)),
                )

    records = {}
    for key in set(received) | set(started) | set(kv_ready) | set(completed):
        kv = kv_ready.get(key)
        records[key] = {
            "worker": worker_label,
            "request_id": key,
            "t_received": received.get(key),
            "t_started": started.get(key),
            "t_kv_ready": kv[0] if kv else None,
            "compute_ms": kv[1] if kv else None,
            "t_completed": completed.get(key),
        }
    return records


def parse_decode_log(path, year):
    received, completed, xfer = {}, {}, {}
    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            m = RECEIVED_RE.search(line)
            if m:
                received[m.group(3)[:8]] = parse_iso(m.group(1), m.group(2))
                continue
            m = COMPLETED_RE.search(line)
            if m:
                completed[m.group(3)[:8]] = parse_iso(m.group(1), m.group(2))
                continue
            m = XFER_RE.search(line)
            if m:
                t_end = parse_short(m.group(1), m.group(2), year)
                elapsed_ms = float(m.group(3))
                xfer[m.group(4)[:8]] = (t_end - elapsed_ms / 1000.0, t_end, elapsed_ms)

    return received, completed, xfer


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "logs", nargs="+", help="worker_label=path pairs, e.g. worker0=/path/to/log"
    )
    ap.add_argument("--decode", help="path to the RDU decode worker's log")
    ap.add_argument(
        "--year",
        default="2026",
        help="year prefix for the connector's MM-DD HH:MM:SS timestamps",
    )
    ap.add_argument(
        "--start", type=float, help="drop requests received before this unix ts"
    )
    ap.add_argument(
        "--end", type=float, help="drop requests received after this unix ts"
    )
    ap.add_argument("-o", "--output", help="output JSON path (default: stdout)")
    args = ap.parse_args()

    all_records = {}
    for spec in args.logs:
        if "=" not in spec:
            print(f"ERROR: expected worker_label=path, got {spec}", file=sys.stderr)
            sys.exit(1)
        label, path = spec.split("=", 1)
        all_records.update(parse_prefill_log(path, label, args.year))

    if args.decode:
        d_received, d_completed, d_xfer = parse_decode_log(args.decode, args.year)
        for key in set(d_received) | set(d_completed) | set(d_xfer):
            rec = all_records.setdefault(
                key,
                {
                    "worker": None,
                    "request_id": key,
                    "t_received": None,
                    "t_started": None,
                    "t_kv_ready": None,
                    "compute_ms": None,
                    "t_completed": None,
                },
            )
            rec["t_received_decode"] = d_received.get(key)
            rec["t_completed_decode"] = d_completed.get(key)
            xf = d_xfer.get(key)
            rec["t_xfer_start"] = xf[0] if xf else None
            rec["t_xfer_end"] = xf[1] if xf else None
            rec["xfer_elapsed_ms"] = xf[2] if xf else None

    records = list(all_records.values())

    if args.start is not None or args.end is not None:

        def in_window(r):
            t = r["t_received"]
            if t is None:
                return True  # keep undated records rather than silently dropping
            if args.start is not None and t < args.start - 10:
                return False
            if args.end is not None and t > args.end + 10:
                return False
            return True

        records = [r for r in records if in_window(r)]

    records.sort(key=lambda r: (r["t_received"] or 0))
    out = json.dumps(records, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {len(records)} request records -> {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
