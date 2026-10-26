# Phase 0a physical M3 Max: nonempty FP16 RATE capture and encoder counters

**Date:** 2026-10-26  
**Evidence status:** physical-M3-Max Xcode screenshots and user-supplied `Export Encoder Counters` CSV; **not** hosted macOS CI, **not** an M5 Max benchmark.  
**Capture revision:** [PR #4](https://github.com/kaaburgh/apple-matrix-capabilities/pull/4), commit `bff9f6bf5cb3b8c38f1ce2ed7eac8d2201bd6236`, fresh-stream capture mode.  
**Local trace label:** `phase0a-f16-rate-fix-bff9f6b-001` (raw trace remains private).  
**Platform:** physical Apple M3 Max; macOS 26.7 (25G229); Xcode 27.0 (27A266a). The local environment is not asserted to be equivalent to the published M5 Max SKU.

## What was verified in Xcode

| Property | Observation |
| --- | --- |
| Command buffers | **1** |
| Compute encoders | **1** |
| Dispatch calls | **1** |
| Target function | `custom_kernel_amc_f16_f16__f32_2048x4096x4096_64_128_4_float16_t_float16_t_float` |
| Shader identity | Contains expected `amc_f16_f16__f32_2048x4096x4096_64_128_4` substring |
| Xcode Performance → Effective GPU Time | **5.30 ms** |
| Performance state | Maximum |
| SIMD groups | 4,136 |
| Allocated registers | 192 |
| Spilled bytes | 96 |
| GPU buffers displayed | 80.00 MiB |

This resolves the **original empty-capture bug** on the physical M3 Max: the initial trace had **0 command buffers / 0 dispatches**; the corrected capture has exactly one intended FP16 RATE dispatch. Xcode Performance and Counters views populated successfully. An independent ordinary replay *without* profiling was not documented separately.

## Encoder counters: selected original CSV fields

The operator exported **Performance → Counters → Export Encoder Counters** in Xcode. This is **one encoder row** with 253 original CSV columns: 5 identifying/blank columns and **248 numerical counter columns**, 61 of them nonzero. Counter names are copied **exactly** from the original export; values below are the raw displayed numbers. Percentage interpretation follows the corresponding Xcode UI headings (the CSV itself does not carry explicit units for every field).

| Original counter name | CSV value |
| --- | ---: |
| `F32 Utilization` | 91.58 |
| `F32 Limiter` | 96.04 |
| `F16 Utilization` | 0.00 |
| `F16 Limiter` | 0.00 |
| `Instruction Throughput Limiter` | 90.52 |
| `Instruction Throughput Utilization` | 23.82 |
| `ALU Utilization` | 46.24 |
| `Kernel Occupancy` | 24.10 |
| `Kernel ALU Instructions` | 34,693,227,328.00 |
| `Kernel ALU Float Instructions` | 99.04 |
| `Last Level Cache Miss Rate` | 10.44 |

The source CSV has **no column whose name contains `Neural` or `Tensor`**. This means those counters were not present *by that name in this M3 Max export*, **not** that such counters are proven unavailable on future M5 Max hardware.

**Provenance check:** SHA-256 of the original local CSV `phase0a-f16-rate-fix-bff9f6b-001 Counters.csv`:
`ab9f73c359b398a76d98ba280721b20d793cea08a0638b829eb8298cda7f399a`.

The original 253-column CSV and raw `.gputrace` were **not committed**. The CSV's one-row nature and repeated counter labels in some unrelated fields mean that future bulk comparisons should preserve original counter indices, not merely normalize by name.

## Interpretation limits and next work

- These values establish that **Xcode can collect per-encoder GPU performance counters** for the captured FP16 RATE workload on a physical M3 Max. They also suggest a heavily F32-limited arithmetic path, which merits controlled follow-up; **they do not prove any particular GPU ISA, data conversion pathway, or Neural Accelerator use/absence**.
- `Kernel ALU Instructions` is a profiler-defined metric, not automatically a count of mathematical multiply-accumulate operations. Do not identify it 1:1 with the algebraic GEMM operation count without validation.
- **5.30 ms is an Xcode capture/replay performance observation**, not a steady-state, independently reproduced Phase 0b timing result; do **not** insert it into `receipts/`, use it for a normalized M3-vs-M5 comparison or select batch N from it.
- M5 Max counter availability, names and scope must be determined **on that physical machine**. Maintain equivalent lane, shape, tile, precision, power/toolchain provenance, and compare supported counter semantics rather than assuming matching columns.
- Confirm standalone ordinary replay separately if needed, and track potential `Profile after replay` residency-set issues without equating them with the original capture failure.

Evidence summary for the capture-plumbing fix; the research protocol and M5 Max gate remain in [issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1).
