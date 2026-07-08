#!/usr/bin/env python3
"""Parse prefill/decode worker log(s) for per-request timing events.

Extracts, per request_id (matched on the first 8 hex chars — the connector's
own log lines truncate the UUID to that length):
  - dynamo-ingress `request received` / `request completed` (UTC, sub-second)
  - the NIXL connector's `prefill.req.started` / `kv_ready ...
    prefill_elapsed_ms=N` (UTC, second-granularity only — that log format
    has no sub-second timestamp)

Output: a JSON list of records, one per request, with unix-epoch-UTC (float)
timestamps:
  {worker, request_id, t_received, t_started, t_kv_ready, compute_ms, t_completed}

Usage:
  python3 extract_request_events.py worker0=/path/to/gpu_prefill_0.log \\
      worker1=/path/to/gpu_prefill_1.log \\
      --start 1783486038.86 --end 1783486398.19 -o events.json

Note: the connector's `prefill.req.started`/`kv_ready` lines have no year in
their timestamp (`MM-DD HH:MM:SS`) — pass --year if analyzing logs from a
year other than the default.
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


def parse_iso(date_s, time_s):
    dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def parse_short(mmdd, hms, year):
    dt = datetime.strptime(f"{year}-{mmdd} {hms}", "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def parse_log(path, worker_label, year):
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

    records = []
    for key in set(received) | set(started) | set(kv_ready) | set(completed):
        kv = kv_ready.get(key)
        records.append(
            {
                "worker": worker_label,
                "request_id": key,
                "t_received": received.get(key),
                "t_started": started.get(key),
                "t_kv_ready": kv[0] if kv else None,
                "compute_ms": kv[1] if kv else None,
                "t_completed": completed.get(key),
            }
        )
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "logs", nargs="+", help="worker_label=path pairs, e.g. worker0=/path/to/log"
    )
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

    all_records = []
    for spec in args.logs:
        if "=" not in spec:
            print(f"ERROR: expected worker_label=path, got {spec}", file=sys.stderr)
            sys.exit(1)
        label, path = spec.split("=", 1)
        all_records.extend(parse_log(path, label, args.year))

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

        all_records = [r for r in all_records if in_window(r)]

    all_records.sort(key=lambda r: (r["t_received"] or 0))
    out = json.dumps(all_records, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {len(all_records)} request records -> {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
