# PR A: reusable float-lane capture and Instruments live loop

This implements **the reusable code portion** of [issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1), building on merged Phase 0a (physical M3 Max capture verified) and Phase 0b (uncaptured baseline methodology).

**Hardware/research gates remain open:** physical M5 Max correctness, replay settings, Instruments counter scope and GPU idle fraction, and a ±5% or overlapping-uncertainty FP16 Phase 0b cross-check. CI/CPU mocks are not substitutes. This PR does not create receipts or modify existing benchmark defaults.

## Supported existing lane names

`f16*f16->f32`, `e4m3*e4m3->f32`, `fp4*fp4->f32`, `f32*f32->f32`, and `strict_f32*f32->f32`.

The runner imports the **stock** `probes/lane_sweep.py`. It uses the existing upstream `LANES`, unchanged seed indices, `check_tile()`, kernel source, TensorOps layout, strict/relaxed precision, and packing discovery (try low nibble first; high nibble **only for packed FP4 after a failed CHK**). It cannot expand the meaning of the upstream **intended-value correctness** test: independently decoded format semantics remain unverified. The original import intentionally applies `cap_memory()`; record the documented limit.

`--shape rate` uses upstream RATE = (2048,4096,4096) and first checks CHK = (64,1024,256). `--shape check` is for quick setup and CI **only**. `--tile 0` is (64,128,4); other tile indices are taken from `lane_sweep.TILES`. Correctness and NumPy reference run *before* either capture or timed looping. RATE failure never triggers another packing choice. This runner resets the deterministic RNG for each CHK nibble attempt and does not iterate the earlier RATE tiles; input bytes are **not identical** to a full upstream sweep's RNG progression.

## Installation and local experiment paths

```sh
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'mlx==0.32.3' 'numpy>=2,<3'
mkdir -p experiments/m3-max
```

Xcode 27 with Metal SDK is required for capture. Compare against the published M5 Max reference only with explicit macOS/Xcode/Metal/MLX and chip-SKU provenance; macOS 26.7 / Xcode 27.0 is the verified physical M3 Max combination.

## Capture — one bounded dispatch and no timing claim

```sh
MTL_CAPTURE_ENABLED=1 python probes/capture_lane.py \
  --lane 'f16*f16->f32' --shape rate --tile 0 \
  --capture experiments/m3-max/f16-rate-001.gputrace \
  --output experiments/m3-max/f16-rate-001.json

open experiments/m3-max/f16-rate-001.gputrace
```

This uses the **Phase 0a fresh GPU stream constructed after start_capture** and synchronous MLX evaluation. It does not add an extra trailing synchronize/empty command buffer. For a possible command-queue residency-set replay issue, `--capture-mode default-stream` preflushes the default command buffer inside capture *before* dispatch. The fallback remains experimental on physical hardware.

Default `--batch 1` is intentional for a single GPU command. A bounded `--batch N` can capture N independent outputs with one synchronized evaluation; **expect exactly N dispatch calls, but one or more command buffers** depending on MLX command-buffer limits. Outputs may overlap rather than serialize. Verify in Xcode, never from trace size. The manifest stores the expected shader substring, requested N, source hashes, lane layout, precision, allocator/memory guard and toolchain. It deliberately does **not** contain an uncaptured `converged` field. A written trace is still marked *replay not verified* until inspected.

For each tested lane, record **Xcode replay settings and its observed command buffers, expected kernel name and dispatch calls** in a separate sanitized note. Compare Xcode replay timings only against same-configuration replays, not against live throughput.

## Live loop — uncaptured GPU load for Instruments

```sh
env -u MTL_CAPTURE_ENABLED -u MTL_SHADER_VALIDATION -u MTL_DEBUG_LAYER \
  -u METAL_DEVICE_WRAPPER_TYPE \
  python probes/capture_lane.py \
    --lane 'f16*f16->f32' --shape rate --tile 0 --batch 8 \
    --loop 30 --output experiments/m3-max/f16-loop-001.json
```

**N=8 is a provisional option, not a chosen optimal M5 Max batch size.**
Capture/debug/shader-validation/device-wrapper modes are rejected. Bounded warm-up uses the same Phase 0b per-batch, last-three-window convergence criterion; default ≥0.5 s and ≤8 s, last-three spread ≤2%. Whether convergence succeeds is explicitly recorded; failure makes loop output **diagnostic only**, not a steady-state measurement.

`--loop` constructs N independent lazy MLX dispatches per `mx.eval` synchronization, reports separate construction and eval+sync times and guards N × output bytes (default ≤512 MiB and ≤10% of detected RAM). A 250 ms quiet gap occurs **before warm-up**, followed by continuous warm-up and loop without a gap, then a final 250 ms gap. Full-loop stability examines every populated time window (including initial ramp and late throttling); only both stable loop and converged warm-up permit a steady-state claim. The full batch time series and window settings remain local. **GPU idle fraction and Neural Accelerator utilization are null, not guessed**: inspect the matching window in Metal System Trace, record fraction of GPU idle time next to utilization and include scope/sampling granularity. A live loop is **not** Xcode capture replay.

## Recording a Metal System Trace

Confirm the template name and availability on the **physical target Mac** (not validated by hosted CI):

```sh
xcrun xctrace list templates
```

If the list contains `Metal System Trace`, record an **uncaptured** workload with a fresh local trace path:

```sh
xcrun xctrace record --template 'Metal System Trace' \
  --output experiments/m3-max/f16-loop-001.trace --time-limit 60s \
  --launch -- /usr/bin/env -u MTL_CAPTURE_ENABLED -u MTL_SHADER_VALIDATION \
     -u MTL_DEBUG_LAYER -u METAL_DEVICE_WRAPPER_TYPE \
     .venv/bin/python probes/capture_lane.py --lane 'f16*f16->f32' \
     --shape rate --tile 0 --batch 8 --loop 30 \
     --output experiments/m3-max/f16-loop-001.json
```

Open the resulting `.trace` in Instruments. Find **idle gap → warm-up
(`warmup.elapsed_s`) → uninterrupted measured loop → idle gap**. The
`loop.samples[].offset_s` timestamps are relative to the end of warm-up.
Count the GPU-idle fraction in the selected loop segment, document its scope and
sampling granularity, and compare only with like-for-like sessions.

**Privacy:** Instruments `.trace` contains process names and system metadata,
often unrelated to the target workload. Keep it local like `.gputrace`;
publish only reviewed and sanitized derived observations.

## Phase 0b comparator and module maintenance decision

For the first controlled **matched-format comparison**, use common tile
**64×64×4 (`--tile 2`)** for FP16, E4M3, FP4 and both F32 controls
where supported. `--tile 0` in the quick-start example is only a
reference to the existing Phase 0a RATE geometry; **matched-tile
comparisons are not best-tile rankings**.

**Decision (reuse model b):** Phase 0a capture lifecycle and Phase 0b
warm-up/T(N) primitives are now **maintained dependencies** of PR A, no longer
disposable implementations. Keep the independently frozen Phase 0b baseline
source SHA and measurements immutable; any later refactor gets a new SHA and
requires new evidence. The source hashes in each PR A manifest include both
dependency files. Older docs using 'disposable' describe the original
experiment scope, not permission to delete these code paths.

A single-N loop cannot estimate Phase 0b's fitted T(N) slope `b`.
The only like-for-like host-clock comparison is the **stable loop median
`median_eval_ms_per_dispatch_stable_only`** against Phase 0b
`batch_timing.per_n["N"].eval_ms.median / N` on the same shape, tile,
SKU, toolchain and batch N. The old aggregate `host_eval_dispatches_per_s`
mixed ramp and fixed overhead and is no longer a benchmark metric.
The `b` cross-check must separately rerun the source-pinned Phase 0b
multi-N experiment and compare with a **different measurement surface**,
such as the selected Instruments GPU-busy interval, where actual hardware
GPU work and idle time are observable. Shared MLX timing implementation alone
shows reproducibility, not independent validation. The ±5% gate is **not
satisfied** by CPU tests or this PR.

## Required controls and decisions

Run FP16 first, then one existing E4M3 and one FP4 example, and both relaxed/strict f32 controls. Use **64×64×4 (`--tile 2`)** for the matched FP32 and format contrasts and report whether relaxed/strict paths agree on the upstream intended-value inputs. Differences in F32/NA counters do not establish physical low-bit ISA instructions.

Real-device manual gate (separate from code merge):
- Verify one actual target dispatch in an Xcode capture per selected lane, identifiable `amc_...` function and documented replay settings.
- Record Xcode counter names, scopes, availability, Performance Limiters and Neural Accelerator utilization only when calibrated. Zero is not the same as unavailable.
- Inspect loop window and GPU idle fraction in Instruments; report nonconvergence without selecting a steady-state tile.
- On the **same target M5 Max SKU**, compare FP16 loop results to the frozen Phase 0b reference: within predeclared ±5% or overlapping repeat uncertainty, otherwise investigate.
- Keep raw `.gputrace` / `.trace` / local manifest files and machine-private diagnostics out of public commits; publish a reviewed derived summary in issue #1.

Do not use `receipts/` or `matrix/` for these experimental manifests. The script requires a **new** output filename and uses Phase 0b atomic no-overwrite publishing.
