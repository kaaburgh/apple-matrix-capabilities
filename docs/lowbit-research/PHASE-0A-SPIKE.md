# Phase 0a: one-dispatch FP16 Metal capture spike

This **small, disposable manual experiment** implements only the first hardware feasibility step from [PLAN.md](PLAN.md) / [PHASE-0.md](PHASE-0.md), tracked by [issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1).

**Not a benchmark.** No automatic Neural Accelerator counters, ISA claims or steady-state rates. Linux CI checks only syntax and synthetic guard paths; **no Apple hardware result has been produced by this PR**.

## On a real Apple Silicon Mac

Prerequisites: a supported Apple GPU, full Xcode **27 or newer** (verify `xcodebuild -version`), the matching Metal SDK, and Python 3.12 or another compatible Python. For comparisons with Precisit's October 2026 receipts, use exactly `mlx==0.32.3`. All commands below run locally; don't upload trace bundles.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'mlx==0.32.3' numpy

# In the repository root; pick an unused path (never overwrite):
MTL_CAPTURE_ENABLED=1 python tools/phase0a_capture_spike.py \
  --trace phase0a-f16.gputrace

open phase0a-f16.gputrace
```

The script reuses the unmodified `lane_sweep.py` FP16 lane, its **64×1024×256** correctness shape and **64×128×4** tile. It validates against the upstream NumPy reference, warms once outside capture and captures exactly one additional `mpp::tensor_ops::matmul2d` dispatch via stock MLX. On success it prints a **local-only** provenance JSON summary and creates a trace bundle.

The successful command means **only** that the trace bundle was created. An operator must verify in Xcode that it opens and replays, contains the expected `amc_...` kernel dispatch, and has useful profiler information. Document the *Xcode replay settings*. If the stock MLX wheel lacks source attribution, optionally use `-DMLX_METAL_DEBUG=ON` on the **same stock MLX tag**, recording the difference.

## Diagnostic GitHub Actions (not M5 Max profiling)

[Phase 0a macOS Metal diagnostic](../../.github/workflows/phase0a-macos.yml)
runs for relevant PR changes and can also be dispatched manually after the
workflow exists on the default branch. It compares GitHub-hosted
`macos-26` and `xcode-27` **ARM64 M1 virtual machines**.

- It locates an installed full Xcode 27, checks `xcrun metal` **by compiling
  and linking a minimal Metal shader**, and checks that Metal exposes a GPU.
  Some Xcode runner images omit the separately downloadable Metal Toolchain
  even though `xcodebuild -version` succeeds. A missing toolchain or GPU is
  reported as **BLOCKED**, rather than an attempted successful capture.
- When all prerequisites exist, CI installs `mlx==0.32.3` and NumPy,
  performs the single FP16 capture, checks that a trace exists and validates
  the script's status fields. A failed attempted capture **fails the job**.
  **No raw GPU trace, logs, or profiler screenshots are uploaded**;
  temporary local artifacts are deleted at job completion.
- The job summary differentiates **BLOCKED**, **TRACE WRITTEN** and **FAILED**.
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
- [ ] Successfully create/replay an FP16 Metal TensorOps trace on actual Apple hardware.
- [ ] Identify real GPU/NA counters and scope in the relevant Xcode tools.
- [ ] Record source hashes, observed results and blockers in issue #1.

**No claims about availability or performance of native FP4/FP8 instructions are supported by this spike.**
