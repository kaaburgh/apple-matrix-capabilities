# Phase 0a: one-dispatch FP16 Metal capture spike

This **small, disposable manual experiment** implements only the first hardware feasibility step from [PLAN.md](PLAN.md) / [PHASE-0.md](PHASE-0.md), tracked by [issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1).

**Not a benchmark.** No automatic Neural Accelerator counters, ISA claims or steady-state rates. Linux CI checks syntax and CPU stubs. A previous hosted-M1 `xcode-27` Actions run successfully **created** a FP16 trace; that confirms capture plumbing only, not Xcode replay, an M5 Max run, or Neural Accelerator counters.

## On a real Apple Silicon Mac

Prerequisites: a supported Apple GPU, full Xcode **27 or newer** (verify `xcodebuild -version`), the matching Metal SDK, and Python 3.12 or another compatible Python. For comparisons with Precisit's October 2026 receipts, use exactly `mlx==0.32.3`. All commands below run locally; don't upload trace bundles.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'mlx==0.32.3' numpy

# In the repository root; pick an unused path (never overwrite):
MTL_CAPTURE_ENABLED=1 python tools/phase0a_capture_spike.py \
  --shape rate --trace phase0a-f16-rate-v2.gputrace

# Xcode 27 gpudebug: distinguish empty trace from nonempty if supported.
python tools/inspect_gpu_trace.py --trace phase0a-f16-rate-v2.gputrace

open phase0a-f16-rate-v2.gputrace

# Optional quick two-threadgroup plumbing check (not useful NA profiling):
MTL_CAPTURE_ENABLED=1 python tools/phase0a_capture_spike.py \
  --shape check --trace phase0a-f16-check.gputrace
```

The script reuses the unmodified `lane_sweep.py` FP16 lane via `check_tile()`,
with the **64×128×4** tile. Its default `--shape rate` checks the
**64×1024×256** correctness shape and then checks/captures a single
**2048×4096×4096** dispatch. This uses the same sequential RNG draws as
the original check/rate protocol. At the 64×128×4 tile, the rate-shape
kernel launches **1024 threadgroups** instead of two; this is much
more appropriate for the M5 Max counter-feasibility gate.
Both stages use the upstream NumPy reference and therefore the rate run
performs a substantial CPU matrix multiply before capture. `--shape check`
captures the smaller correctness-shape kernel and is the only shape run in
hosted GitHub CI.

A validated invocation and short warm-up occur outside capture. The captured
invocation is evaluated inside `start_capture`/`stop_capture`, then compared
bit-for-bit with a separate **pre-capture** dispatch **after** capture. The
independent reference test is done by `check_tile()`; the later pre-capture
invocation is not independently compared with NumPy.
The script prints a sanitized JSON report on every exit, distinguishing
`blocked` (exit 2), `failed` (exit 1) and
`capture_written_replay_not_verified` (exit 0). A successful trace is
still **not** a verified Xcode replay or hardware counter reading.
`MTL_SHADER_VALIDATION` must be disabled; the status of the other Metal/MLX
debug environment variables is recorded. The manifest records `relaxed_precision` and the exact `expected_kernel_name_contains` substring for locating the Xcode kernel. The offline Metal CLI is recorded as `metal_cli: present|missing`; the JIT capture can work without it, so absence alone does not block the spike. The source hash, compiler, macOS,
device, GPU core count when available, and power metadata follow the upstream
receipt's conventions.

The successful command means **only** that the trace bundle was created. An operator must verify in Xcode that it opens and replays, contains the expected `amc_...` kernel dispatch, and has useful profiler information. Document the *Xcode replay settings*. If the stock MLX wheel lacks source attribution, optionally use `-DMLX_METAL_DEBUG=ON` on the **same stock MLX tag**, recording the difference.

## Diagnostic GitHub Actions (not M5 Max profiling)

[Phase 0a macOS Metal diagnostic](../../.github/workflows/phase0a-macos.yml)
runs for relevant PR changes and can also be dispatched manually after the
workflow exists on the default branch. It compares GitHub-hosted
`macos-26` and `xcode-27` **ARM64 virtual machines**. On the `xcode-27` runner the reported Metal architecture was **`air64_v27`**, a paravirtualized Metal device on an M1 host. This is a plumbing check, not evidence of physical M1 or M5 Max TensorOps behavior.

- It locates an installed full Xcode 27, checks `xcrun metal` **by compiling
  and linking a minimal Metal shader**, and checks that Metal exposes a GPU.
  Some Xcode runner images omit the separately downloadable Metal Toolchain
  even though `xcodebuild -version` succeeds. A missing toolchain or GPU is
  **BLOCKED** on exploratory `macos-26`, but **fails CI** on the previously
  working `xcode-27` runner (expected-ready regression).
- When all prerequisites exist, CI installs `mlx==0.32.3` and NumPy,
  performs the fast `--shape check` FP16 capture, checks that a trace exists, confirms the captured output matches, and validates
  the script's status fields. A failed attempted capture **fails the job**.
  **No raw GPU trace, logs, or profiler screenshots are uploaded**;
  temporary local artifacts are deleted at job completion.
- The job prints only sanitized status/stage/error and environment diagnostics; raw compiler stderr stays local. The job summary differentiates **BLOCKED**, **TRACE WRITTEN** and **FAILED**.
  A green result on a blocked runner is **not evidence of GPU execution**;
  even a written trace has not been opened/replayed in Xcode.
- GitHub-hosted M1 execution cannot establish **M5 Max** Neural Accelerator
  utilization, chip-specific counters, FP4/FP8 native execution, or
  throughput. Those remain subject to the real-target manual gate.
- If no GitHub Actions runs appear after pushing this workflow, verify
  whether Actions are enabled in the fork's repository settings. A workflow
  with `workflow_dispatch` can only be selected manually after the
  workflow file is present on the default branch.

To diagnose a local *missing* Metal Toolchain component, run
`xcrun --sdk macosx --find metal`; if absent, the supported Xcode
installation flow is `xcodebuild -downloadComponent MetalToolchain`.
CI deliberately does **not** mutate the runner's Xcode installation.

## Empty-capture correction (M3 Max, October 2026)

The first physical M3 Max test produced a `.gputrace` that opened in Xcode
but contained **zero command buffers and zero dispatch calls**. The old
`capture_written_replay_not_verified` status and CI file-existence check
never guaranteed trace contents.

MLX 0.32.3's `CommandEncoder` creates a new Metal command buffer ahead of
the next dispatch. If a warmed default stream is reused, capture can commit
a buffer created **before** the capture interval. Apple's capture contract
requires command buffers to be both **created and committed between capture
start and stop**. The revised experiment therefore:

1. Calls `mx.metal.start_capture()`.
2. Creates `mx.new_stream(mx.gpu)` **within** the capture interval.
3. Dispatches using `with mx.stream(capture_stream)`, then `mx.eval(result)`
   and `mx.synchronize(capture_stream)` before stopping capture.
4. Calls `stop_capture()` in `finally`.

`mx.eval()` is already synchronous in upstream MLX. Synchronization is
additional defensive sequencing, **not a proven explanation of the failure**.
The new stream and its freshly created Metal command buffer are the material
change. The mechanism is an evidence-backed hypothesis until the corrected
M3 Max capture is opened and inspected.

Run `tools/inspect_gpu_trace.py` after capture. Where Xcode 27's
`gpudebug` works, it returns exit 0 for **nonempty** command buffers,
exit 1 for an **empty** capture, and exit 2 for **unverified** (tool
unavailable, unsupported output, etc.). It prints only sanitized counts;
a trace bundle merely existing is not a positive result. For a completed
feasibility check, verify **Command Buffers > 0** and **Dispatch Calls > 0**
in Xcode, identify the expected `amc_...` dispatch and report the Xcode
replay/counter status. The gpudebug CLI's buffer count alone does not prove
the intended kernel is present.

Re-run with a **new** path; do not overwrite the empty trace, which is
useful local diagnostic evidence. Keep all raw traces on the Mac and
do not attach them to public issues.

## Follow-up: counter feasibility (manual, not implemented here)

1. Inspect **Metal Debugger** for per-shader statistics and instruction mix, if available.
2. Run a separate **Instruments Metal System Trace** session; identify actual Neural Accelerator / Performance Limiters counters and their granularity.
3. Validate against an FP16 TensorOps positive control and independently sourced `simdgroup_matrix` negative control (with correct reference outputs and source SHA). The negative-control kernel is **not** part of this initial PR.
4. If practical, test whether `xctrace` exports the actual desired counters. Never assume `.gputrace` and Instruments `.trace` are interchangeable.
5. Post **sanitized observations** and explicit blocker status to issue #1, including chip/SKU, Xcode build, MLX version, source SHA and replay conditions. No raw GPU captures, private paths, process listings or unredacted screenshots.

## Acceptance and current status

- [x] A CPU-testable capture script exists without importing MLX on Linux.
- [x] Input validation refuses missing capture opt-in, existing paths and non-Apple Silicon.
- [x] Capture shutdown is protected by `finally`, including error paths.
- [x] A hosted-M1 GitHub Actions run created a **check-shape** FP16 trace (without replay).
- [ ] Successfully create **and replay** a rate-shape FP16 Metal TensorOps trace on a target M5 Max.
- [ ] Identify real GPU/NA counters and scope in the relevant Xcode tools.
- [ ] Record source hashes, observed results and blockers in issue #1.

**No claims about availability or performance of native FP4/FP8 instructions are supported by this spike.**
