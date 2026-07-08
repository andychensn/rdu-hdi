#!/usr/bin/env python3
"""Plot nvidia-smi GPU telemetry (from launch/gpu_prefill.sh's background
sampler) with per-request event annotations overlaid.

nvidia-smi's own `timestamp` field is in the GPU node's LOCAL time (PDT on
this cluster), while the event JSON from extract_request_events.py is UTC —
--tz-offset-hours converts local -> UTC (default 7, i.e. PDT). Adjust for
PST (8) or other timezones/seasons.

Usage:
  python3 plot_request_timeline.py \\
      --smi worker0=/path/to/gpu_telemetry_0_job123.csv \\
      --smi worker1=/path/to/gpu_telemetry_1_job456.csv \\
      --gpus worker0=4,5,6,7 --gpus worker1=0,1,2,3 \\
      --events events.json \\
      --start 1783486038.86 --end 1783486398.19 \\
      -o timeline.png --title "100k/1k conc=1, 20 requests"
"""
import argparse
import re
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

try:
    import matplotlib
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIT_RE = re.compile(r"[^\d.\-]")


def strip_unit(s):
    v = UNIT_RE.sub("", s.strip())
    return float(v) if v not in ("", "-") else None


def parse_smi_csv(path, tz_offset_hours):
    """nvidia-smi -lms output: timestamp,index,util,power,clock,temp (local node time)."""
    rows = defaultdict(list)  # gpu_index -> [(unix_ts_utc, util, power, clock, temp), ...]
    with open(path, errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 6:
                continue
            try:
                dt = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                continue
            dt_utc = dt.replace(tzinfo=timezone.utc) + timedelta(hours=tz_offset_hours)
            ts = dt_utc.timestamp()
            try:
                gpu = int(parts[1])
            except ValueError:
                continue
            rows[gpu].append(
                (ts, strip_unit(parts[2]), strip_unit(parts[3]), strip_unit(parts[4]), strip_unit(parts[5]))
            )
    return rows


def group_series(rows, gpu_indices, field_idx):
    """Average `field_idx` (1=util,2=power,3=clock,4=temp) across gpu_indices per shared timestamp."""
    by_ts = defaultdict(list)
    for g in gpu_indices:
        for rec in rows.get(g, []):
            if rec[field_idx] is not None:
                by_ts[rec[0]].append(rec[field_idx])
    ts_sorted = sorted(by_ts)
    vals = [sum(by_ts[t]) / len(by_ts[t]) for t in ts_sorted]
    return ts_sorted, vals


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--smi", action="append", required=True, help="worker_label=path.csv (repeatable)")
    ap.add_argument("--gpus", action="append", required=True, help="worker_label=0,1,2,3 (repeatable)")
    ap.add_argument("--events", required=True, help="JSON from extract_request_events.py")
    ap.add_argument("--start", type=float, required=True, help="benchmark start, unix epoch UTC")
    ap.add_argument("--end", type=float, required=True, help="benchmark end, unix epoch UTC")
    ap.add_argument(
        "--tz-offset-hours",
        type=float,
        default=7.0,
        help="hours to ADD to nvidia-smi's local timestamp to get UTC (default 7 = PDT)",
    )
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    gpu_map = {}
    for spec in args.gpus:
        label, idxs = spec.split("=", 1)
        gpu_map[label] = [int(x) for x in idxs.split(",")]

    smi_map = {}
    for spec in args.smi:
        label, path = spec.split("=", 1)
        smi_map[label] = parse_smi_csv(path, args.tz_offset_hours)

    events = json.load(open(args.events))

    palette = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad"]
    color_by_worker = {label: palette[i % len(palette)] for i, label in enumerate(smi_map)}
    QUEUE_COLOR = "#bdc3c7"
    XFER_COLOR = "#f39c12"

    fig, axes = plt.subplots(
        5, 1, figsize=(18, 14), sharex=True,
        gridspec_kw={"hspace": 0.15, "height_ratios": [1.3, 1, 1, 1, 1]},
    )
    fig.suptitle(
        f"Per-request phase timeline + GPU telemetry{(' — ' + args.title) if args.title else ''}",
        fontsize=12,
        fontweight="bold",
    )

    # ── Panel 0: per-request phase Gantt (queue -> prefill -> KV transfer) ──
    gantt_ax = axes[0]
    workers = list(smi_map.keys())
    row_y = {w: i for i, w in enumerate(workers)}
    bar_h = 0.6
    has_xfer = False
    for rec in events:
        worker = rec.get("worker")
        if worker not in row_y:
            continue
        y = row_y[worker] - bar_h / 2
        t_recv = rec.get("t_received")
        t_start = rec.get("t_started")
        t_kv = rec.get("t_kv_ready")
        t_xfer_s = rec.get("t_xfer_start")
        t_xfer_e = rec.get("t_xfer_end")
        segs = []
        if t_recv is not None and t_start is not None:
            segs.append((t_recv - args.start, t_start - t_recv, QUEUE_COLOR))
        if t_start is not None and t_kv is not None:
            segs.append((t_start - args.start, t_kv - t_start, color_by_worker[worker]))
        if t_xfer_s is not None and t_xfer_e is not None:
            segs.append((t_xfer_s - args.start, t_xfer_e - t_xfer_s, XFER_COLOR))
            has_xfer = True
        if segs:
            gantt_ax.broken_barh([(s[0], s[1]) for s in segs], (y, bar_h),
                                  facecolors=[s[2] for s in segs], edgecolor="none")

    gantt_ax.set_yticks(list(row_y.values()))
    gantt_ax.set_yticklabels(workers, fontsize=9)
    gantt_ax.set_ylim(-1, len(workers))
    gantt_ax.set_ylabel("Request phases", fontsize=9)
    gantt_ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    gantt_ax.spines["top"].set_visible(False)
    gantt_ax.spines["right"].set_visible(False)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=QUEUE_COLOR, label="queue (arrived, waiting)"),
        plt.Rectangle((0, 0), 1, 1, color="#7f8c8d", label="prefill compute"),
    ]
    if has_xfer:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=XFER_COLOR, label="KV transfer"))
    else:
        gantt_ax.annotate(
            "(no KV-transfer segments found -- decode log needs VLLM_RDU_PLUGIN_TIME_PROFILE=1)",
            xy=(0.5, 1.02), xycoords="axes fraction", ha="center", fontsize=7, color="#999",
        )
    gantt_ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)

    panels = [
        (1, "GPU Utilization (%)", (0, 108)),
        (2, "Power (W)", None),
        (3, "SM Clock (MHz)", None),
        (4, "Temperature (C)", None),
    ]

    for ax, (field_idx, ylabel, ylim) in zip(axes[1:], panels):
        for label, rows in smi_map.items():
            ts, vals = group_series(rows, gpu_map[label], field_idx)
            t_rel = [t - args.start for t in ts]
            ax.plot(t_rel, vals, color=color_by_worker[label], linewidth=0.8, alpha=0.9, label=label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Per-request shaded spans + labels, on the utilization panel only (else too busy)
    util_ax = axes[1]
    per_worker_counts = defaultdict(int)
    spans = []
    for rec in events:
        worker = rec.get("worker")
        if worker not in color_by_worker:
            continue
        t0 = rec.get("t_received")
        t1 = rec.get("t_kv_ready") or rec.get("t_completed")
        if t0 is None or t1 is None:
            continue
        per_worker_counts[worker] += 1
        idx = per_worker_counts[worker]
        util_ax.axvspan(t0 - args.start, t1 - args.start, color=color_by_worker[worker], alpha=0.12, lw=0)
        spans.append((t0, t1, worker, idx, rec.get("compute_ms")))

    # Stagger labels across a few vertical levels by GLOBAL time order (not
    # per-worker) so adjacent bumps close in time don't overlap regardless of
    # which worker they belong to.
    spans.sort(key=lambda s: s[0])
    levels = [104, 116, 128]
    for i, (t0, t1, worker, idx, compute_ms) in enumerate(spans):
        c = color_by_worker[worker]
        label_txt = f"{worker}#{idx}" + (f" {compute_ms}ms" if compute_ms else "")
        util_ax.annotate(
            label_txt,
            xy=((t0 + t1) / 2 - args.start, levels[i % len(levels)]),
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=60,
            color=c,
            annotation_clip=False,
        )

    util_ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    axes[-1].set_xlabel("Time (seconds from benchmark start)", fontsize=10)
    util_ax.set_ylim(0, 145)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Chart saved -> {args.output}")


if __name__ == "__main__":
    main()
