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

Any number of --smi/--gpus worker pairs is supported (not just 2) -- each
worker's telemetry and physical-GPU-index mapping is independent, so this
works the same whether all workers share one node or are spread across
several (each worker's --gpus indices are local to its own --smi CSV, not a
shared/global GPU index space).

--metrics controls which telemetry panel(s) render below the Gantt chart,
comma-separated from {util,power,clock,temp} -- default is just "util" (one
line per worker, averaged across that worker's own GPUs) to keep the chart
readable at higher worker counts; pass e.g. --metrics util,power,clock,temp
to get the original full 4-panel telemetry stack back.
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


def parse_smi_csv(path, tz_offset_hours, start=None, end=None, pad_seconds=3.0):
    """nvidia-smi -lms output: timestamp,index,util,power,clock,temp (local node time).

    These CSVs accumulate for the worker's entire job lifetime (often much
    longer than any single benchmark run) -- filter to [start-pad, end+pad]
    so a short benchmark isn't squeezed into a sliver of a much wider axis.
    Keep this small (a few seconds) and rely on the explicit set_xlim() in
    main() for the actual displayed window -- a large pad here plus
    matplotlib's default autoscaling is what used to leave large empty
    margins on both sides of a short, tightly-windowed benchmark.
    """
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
            if start is not None and ts < start - pad_seconds:
                continue
            if end is not None and ts > end + pad_seconds:
                continue
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
    ap.add_argument(
        "--node",
        action="append",
        default=[],
        help="worker_label=hostname (repeatable, optional) -- rendered as a worker/GPU/node key "
             "under the title so a reader doesn't have to guess which physical GPUs/node each "
             "worker label refers to. Safe to omit if all workers are on one node.",
    )
    ap.add_argument("--events", required=True, help="JSON from extract_request_events.py")
    ap.add_argument("--start", type=float, required=True, help="benchmark start, unix epoch UTC")
    ap.add_argument("--end", type=float, required=True, help="benchmark end, unix epoch UTC")
    ap.add_argument(
        "--tz-offset-hours",
        type=float,
        default=7.0,
        help="hours to ADD to nvidia-smi's local timestamp to get UTC (default 7 = PDT)",
    )
    ap.add_argument(
        "--metrics",
        default="util",
        help="comma-separated telemetry panels to render, from {util,power,clock,temp} "
             "(default: util only, one averaged-per-worker line each -- pass e.g. "
             "util,power,clock,temp for the full original 4-panel stack)",
    )
    ap.add_argument(
        "--pad-seconds",
        type=float,
        default=3.0,
        help="seconds of context to show before --start and after --end (default 3.0). "
             "The displayed window is always exactly [start-pad, end+pad], regardless of "
             "how far the underlying telemetry CSV data actually extends -- pass --start/"
             "--end tightly around the real request activity (e.g. min t_received / max "
             "t_completed_decode from the events JSON) for a tight chart with minimal "
             "empty margin.",
    )
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    ALL_METRICS = {"util": (1, "GPU Utilization (%)", (0, 108)),
                    "power": (2, "Power (W)", None),
                    "clock": (3, "SM Clock (MHz)", None),
                    "temp": (4, "Temperature (C)", None)}
    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = set(requested_metrics) - set(ALL_METRICS)
    if unknown:
        sys.exit(f"--metrics: unknown metric(s) {sorted(unknown)}, choose from {sorted(ALL_METRICS)}")

    gpu_map = {}
    for spec in args.gpus:
        label, idxs = spec.split("=", 1)
        gpu_map[label] = [int(x) for x in idxs.split(",")]

    node_map = {}
    for spec in args.node:
        label, hostname = spec.split("=", 1)
        node_map[label] = hostname

    smi_map = {}
    for spec in args.smi:
        label, path = spec.split("=", 1)
        smi_map[label] = parse_smi_csv(path, args.tz_offset_hours, args.start, args.end,
                                        pad_seconds=args.pad_seconds)

    events = json.load(open(args.events))

    # Worker identity is conveyed by the row label text (e.g. "worker0#3") and
    # the worker/GPU/node key below the title -- NOT by bar fill color. Bar
    # fill color is a fixed 3-color phase legend (queue/prefill/transfer) so
    # the chart always matches its own legend regardless of worker count.
    # color_by_worker still exists for label-text color (a cheap way to
    # visually group a worker's own rows) and for the optional --metrics
    # telemetry panels, which do still plot one line per worker.
    palette = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad"]
    color_by_worker = {label: palette[i % len(palette)] for i, label in enumerate(smi_map)}
    QUEUE_COLOR = "#d5d8dc"
    PREFILL_COLOR = "#2980b9"
    XFER_COLOR = "#f1c40f"

    # One row per REQUEST (not per worker) in the Gantt panel -- at
    # concurrency > 1, multiple requests can be queued/computing on the same
    # worker at once, and cramming them onto one row per worker makes them
    # unreadably overlapped. One row per request instead directly shows how
    # many requests are in flight at any given time, with zero overlap.
    FAILED_COLOR = "#c0392b"
    per_worker_idx = defaultdict(int)
    gantt_rows = []
    has_xfer = False
    n_failed = 0
    for rec in events:
        worker = rec.get("worker")
        if worker not in color_by_worker:
            continue
        t_recv = rec.get("t_received")
        t_start = rec.get("t_started")
        t_kv = rec.get("t_kv_ready")
        t_xfer_s = rec.get("t_xfer_start")
        t_xfer_e = rec.get("t_xfer_end")
        segs = []
        if t_recv is not None and t_start is not None:
            segs.append((t_recv - args.start, t_start - t_recv, QUEUE_COLOR))
        if t_start is not None and t_kv is not None:
            segs.append((t_start - args.start, t_kv - t_start, PREFILL_COLOR))
        if t_xfer_s is not None and t_xfer_e is not None:
            segs.append((t_xfer_s - args.start, t_xfer_e - t_xfer_s, XFER_COLOR))
            has_xfer = True
        if not segs:
            continue
        per_worker_idx[worker] += 1
        row_end = max(s[0] + s[1] for s in segs)
        row_start = min(s[0] for s in segs)
        # Prefill finished (t_kv_ready present) but the request never reached
        # decode (no t_completed_decode) -- a genuine failure/drop, not just
        # "this trial didn't have decode profiling enabled" (which would be
        # uniform across every record, not a subset of them). Distinct from a
        # merely-fast successful request, which would still show real xfer
        # segments if any other record in this same dataset does.
        failed = t_kv is not None and rec.get("t_completed_decode") is None
        if failed:
            n_failed += 1
        label_color = FAILED_COLOR if failed else color_by_worker[worker]
        label_txt = f"{worker}#{per_worker_idx[worker]} {round((row_end - row_start) * 1000)}ms"
        if failed:
            label_txt += " FAILED (no decode completion)"
        gantt_rows.append((t_recv, worker, segs, row_end, label_txt, label_color, failed))

    gantt_rows.sort(key=lambda r: r[0])
    n_rows = len(gantt_rows)
    n_telemetry = len(requested_metrics)

    ROW_IN = 0.26          # inches per request row
    TELEMETRY_IN = 2.6     # inches per telemetry panel
    gantt_in = max(2.0, n_rows * ROW_IN + 0.8)
    fig_h = gantt_in + n_telemetry * TELEMETRY_IN

    fig, axes = plt.subplots(
        1 + n_telemetry, 1, figsize=(18, fig_h), sharex=True,
        gridspec_kw={"hspace": 0.15, "height_ratios": [gantt_in] + [TELEMETRY_IN] * n_telemetry},
    )
    if n_telemetry == 0:
        axes = [axes]  # plt.subplots returns a bare Axes, not an array, for a single row
    telemetry_suffix = " + GPU telemetry" if n_telemetry else ""
    fig.suptitle(
        f"Per-request phase timeline{telemetry_suffix}{(' — ' + args.title) if args.title else ''}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    # Worker/GPU/node key, so a reader doesn't have to guess which physical
    # GPUs (or node) a given worker label refers to -- especially useful once
    # workers are spread across more than one physical node. Positioned as a
    # fixed distance (in inches, via a figure-relative offset scaled by
    # fig_h) below the title rather than a fixed figure-fraction, since the
    # Gantt panel's height (and therefore the whole figure's height) varies
    # a lot with the number of requests plotted -- a fixed fraction like 0.96
    # sits right on top of the title on a short figure and far below it on a
    # tall one.
    worker_key_parts = []
    for label in gpu_map:
        gpus_str = ",".join(str(g) for g in gpu_map[label])
        node_str = f"{node_map[label]} " if label in node_map else ""
        worker_key_parts.append(f"{label}={node_str}GPUs[{gpus_str}]")
    fig.text(0.5, 1 - 0.35 / fig_h, "  |  ".join(worker_key_parts),
              ha="center", fontsize=8, color="#555555")

    # ── Panel 0: per-request phase Gantt (queue -> prefill -> KV transfer), one row per request ──
    gantt_ax = axes[0]
    bar_h = 0.8
    for i, (t_recv, worker, segs, row_end, label_txt, label_color, failed) in enumerate(gantt_rows):
        y = i - bar_h / 2
        gantt_ax.broken_barh(
            [(s[0], s[1]) for s in segs], (y, bar_h),
            facecolors=[s[2] for s in segs],
            edgecolor=FAILED_COLOR if failed else "none",
            linewidth=1.2 if failed else 0,
        )
        gantt_ax.annotate(
            label_txt, xy=(row_end, i), xytext=(4, 0), textcoords="offset points",
            ha="left", va="center", fontsize=5.5,
            fontweight="bold" if failed else "normal",
            color=label_color,
            annotation_clip=False,
        )

    gantt_ax.set_yticks([])
    gantt_ax.set_ylim(-1, n_rows)
    gantt_ax.invert_yaxis()  # earliest-arriving request at the top
    # x-axis is shared across all panels (sharex=True), so this one set_xlim
    # call tightly bounds all of them to [start-pad, end+pad] regardless of
    # how far the underlying telemetry CSVs actually extend -- previously the
    # only bound came from parse_smi_csv's own filtering pad plus matplotlib's
    # default autoscaling, which left large empty margins for a short,
    # tightly-windowed benchmark. annotation_clip=False on the end-of-row
    # labels above still lets them draw past this limit rather than being cut
    # off, so a request ending right at the edge doesn't lose its label.
    gantt_ax.set_xlim(-args.pad_seconds, (args.end - args.start) + args.pad_seconds)
    gantt_ax.set_ylabel(f"Requests ({n_rows}), by arrival", fontsize=9)
    gantt_ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    gantt_ax.spines["top"].set_visible(False)
    gantt_ax.spines["right"].set_visible(False)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=QUEUE_COLOR, label="queue (arrived, waiting)"),
        plt.Rectangle((0, 0), 1, 1, color=PREFILL_COLOR, label="prefill compute"),
    ]
    if has_xfer:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=XFER_COLOR, label="KV transfer"))
    else:
        gantt_ax.annotate(
            "(no KV-transfer segments found -- decode log needs VLLM_RDU_PLUGIN_TIME_PROFILE=1)",
            xy=(0.5, 1.02), xycoords="axes fraction", ha="center", fontsize=7, color="#999",
        )
    if n_failed:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=FAILED_COLOR, linewidth=1.5,
                           label=f"FAILED -- prefill finished, never reached decode ({n_failed})")
        )
    gantt_ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)

    panels = [ALL_METRICS[m] for m in requested_metrics]

    for i, (ax, (field_idx, ylabel, ylim)) in enumerate(zip(axes[1:], panels)):
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
        if i == 0:
            # Worker->color key lives here (once) rather than on every panel --
            # the Gantt legend above explains phases, not which color is which
            # worker.
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    axes[-1].set_xlabel("Time (seconds from benchmark start)", fontsize=10)
    # subplots_adjust (not tight_layout) for the top margin: tight_layout
    # warns it's "not compatible" with this figure (the annotation_clip=False
    # row-end labels aren't standard flow content) and, empirically, leaves
    # a large unpredictable blank gap at the top when there's no telemetry
    # panel to absorb the difference. subplots_adjust's `top` is a direct,
    # unambiguous figure-fraction, so pair it with the same fixed ~0.7in
    # reservation the worker-key text above is positioned relative to.
    fig.subplots_adjust(top=1 - 0.7 / fig_h, left=0.06, right=0.98, bottom=0.05)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Chart saved -> {args.output}")


if __name__ == "__main__":
    main()
