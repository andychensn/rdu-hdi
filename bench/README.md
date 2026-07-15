# bench/ — benchmarking and profiling tools

Two independent tool families live here:

1. **Benchmark runner** (`run.sh`, `sweep.sh`) — drives client traffic against the stack via
   InferenceX's `benchmark_serving.py`.
2. **Profiling / tracing** (everything else below) — turns a benchmark run's logs (and, on the
   RDU side, `coe_api`'s own instrumentation) into a per-request timeline, either as a static PNG
   or as a Chrome Trace Event file viewable in `chrome://tracing` / Perfetto, with GPU prefill
   workers and the RDU decode worker shown side by side on one merged timeline.

---

## Quick reference

| Script | Purpose |
|--------|---------|
| `run.sh` | Run one benchmark config against the live stack |
| `sweep.sh` | Run the standard 9-config benchmark sweep |
| `extract_request_events.py` | Parse prefill + decode worker logs into per-request phase timestamps (JSON) |
| `plot_request_timeline.py` | Static PNG Gantt chart, one panel per GPU prefill worker |
| `events_to_chrome_trace.py` | Convert `extract_request_events.py`'s JSON into Chrome Trace Event Format |
| `merge_traces.py` | Merge N Chrome Trace files (GPU side + RDU decode's `coe_api` dump) into one, with safe `pid` remapping |

---

## Cluster-wide Chrome Trace workflow

This is the interactive alternative to the PNG plot — it shows every GPU prefill worker *and*
the RDU decode worker on one merged timeline, including RDU's rich internal instrumentation
(NIXL transfers, DDR→HBM cache promotion, cache lookups, nova-runtime's own scheduling
internals — tens of thousands of events per request, not just the five coarse request-lifecycle
phases).

**Requires the companion `vllm-rdu` change** (RDUWorker's standalone profiling control server —
see `andyc/hdi-integration` there, or its PR; ported from fast-coe's
`andyc/cluster-profiling-infra` as part of the fast-coe → vllm-rdu migration). Without it, only
the GPU-side trace is available; the steps below just skip the RDU parts.

### 1. (Recommended) Warm up before profiling

The first request into a fresh RDU decode worker pays a one-time NIXL handshake/pipeline setup
cost that would otherwise dominate the trace. Send a couple of small throwaway requests first:

```bash
bash bench/run.sh --input-len 1000 --output-len 10 --concurrency 2 --num-prompts 2
```

### 2. Start RDU-side tracing

The RDU decode worker exposes a tiny HTTP control server on port `9401` (`--net=host`, so it's
reachable directly from the login node at `http://<rdu-node>:9401`):

```bash
curl -X POST http://<rdu-node>:9401/start_profile
```

### 3. Run the real benchmark

```bash
bash bench/run.sh --input-len 100000 --output-len 100 --concurrency 8 --num-prompts 20
```

### 4. Stop RDU-side tracing, with an explicit output path

`/stop_profile` requires a full path in its JSON body — there's no default directory or
auto-generated filename:

```bash
curl -X POST http://<rdu-node>:9401/stop_profile \
    -d '{"path": "/absolute/path/to/rdu_trace.json"}'
```

### 5. Extract GPU-side events and convert to Chrome Trace

```bash
python3 bench/extract_request_events.py \
    worker0=/path/to/gpu_prefill_0.log worker1=/path/to/gpu_prefill_1.log \
    --decode /path/to/rdu_decode.log \
    -o events.json

python3 bench/events_to_chrome_trace.py events.json -o gpu_trace.json
```

### 6. Merge GPU + RDU traces into one file

```bash
python3 bench/merge_traces.py \
    worker0=gpu_trace.json \
    rdu_decode0=/absolute/path/to/rdu_trace.json \
    -o combined_trace.json
```

`merge_traces.py` remaps every source file's `pid` space to fresh, globally-unique values before
concatenating — deliberately not SambaNova's own `snprof -cj`, which does a naive concatenation
with no `pid` remapping and can silently merge two different hosts' processes onto the same
track if their raw OS `pid`s happen to collide.

### 7. View the result

`combined_trace.json` is standard Chrome Trace Event Format:

- Drag-and-drop it into [Perfetto UI](https://ui.perfetto.dev/) or open it via `chrome://tracing`.
- In VS Code (including over Remote-SSH, without copying the file to your local machine): install
  the [Chrome Trace Viewer](https://marketplace.visualstudio.com/items?itemName=jacobdweightman.chrome-trace-viewer)
  extension, right-click the file, **"Open as Profile Trace in Browser."**

Each physical worker is its own labeled process lane (`pid`). GPU-side phases per request:
`queue`, `prefill`, `decode_queue` (prefill done, waiting for a free decode slot — often the
dominant phase under decode-capacity pressure), `kv_transfer`, `decode`. RDU-side events are
`coe_api`'s own — search by event name (e.g. `nixl miss xfer`, `decode admit`,
`stage_copies_ddr2hbm`) or by request ID (embedded in event names/args) rather than trying to
read a specific thread row top-to-bottom; RDU's internal worker-pool threads aren't individually
labeled.

### Known limitations

- **File size**: `coe_api`'s per-request event volume scales with input/prefill token count more
  than with output length — a 100k-input/20-request run produced a combined trace of ~100MB+.
  Large files may load slowly in the viewer; reduce `--num-prompts` or input length if unwieldy.
- **Cross-host timestamp alignment**: durations for the same physical event agree closely between
  the GPU-derived and RDU-native measurements (confirms clocks are sane / NTP-healthy), but
  *absolute* start-time offsets for the same event can vary by hundreds of milliseconds across
  requests — the two sides measure different semantic instants (a log-derived timestamp vs.
  `coe_api`'s actual event-post time), not the same clock reading twice. Trust the merged trace
  for coarse correlation; be more cautious about fine-grained (sub-second) cross-host causal
  ordering claims.
- **Unbounded profiling window**: `coe_api`'s event buffer isn't released until `/stop_profile` is
  called — don't leave `/start_profile` on indefinitely for a long-running server.
