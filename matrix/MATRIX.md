# Capability matrix (generated)

> Generated from [`capabilities.json`](capabilities.json) by [`tools/build_matrix.py`](../tools/build_matrix.py). Do not edit by hand; edit the JSON and run the tool.

Rates are GOPS measured at 2048×4096×4096 (`mode::multiply`), the best result the receipt records: the standard sweep tries five tile configurations and each receipt records the tiles its lanes actually ran. `vs f16` divides the same receipt's `f16*f16->f32` rate, so each chip is compared only with itself. A `⚠` marks a cell whose cited receipt shows a max/min timing spread above 2× across samples: treat it as unstable and rerun before building any dispatch rule on it. These are lane probes, not model numbers. See [METHODS.md](../METHODS.md) and [README.md](../README.md) for the evidence-status model (only observed, receipt-backed rows are published).

## Float lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 1.01× | 1.00× | 1.00× | 1.00× |
| `e4m3*e4m3->f32` | 1.08× | 1.04× | 0.99× ⚠ | **2.08× ⚠** |
| `e5m2*e5m2->f32` | — | 1.04× ⚠ | — | 1.42× |
| `f16*f16->f16` | 0.97× | 1.00× | 0.99× | 1.01× |
| `f16*f16->f32` | 1.00× | 1.00× ⚠ | 1.00× | 1.00× |
| `f32*f32->f32` | 0.50× | 0.56× | 0.62× | 0.48× |
| `fp4*fp4->f32` | 0.93× | 0.93× | 0.93× ⚠ | 1.07× |
| `mx_e4m3*e4m3->f32` | 0.77× | 0.81× | 0.80× ⚠ | 1.02× |
| `mx_e5m2*e5m2->f32` | — | 0.81× | — | 0.97× |
| `mx_fp4*fp4->f32` | 0.71× | 0.73× | 0.73× | 0.76× |
| `relaxed_mx_e4m3*e4m3->f32` | — | 0.82× ⚠ | — | 1.03× |
| `strict_e4m3*e4m3->f32` | — | 1.04× ⚠ | — | **2.07×** |
| `strict_e5m2*e5m2->f32` | — | 1.04× ⚠ | — | 1.42× |
| `strict_f16*f16->f32` | — | 1.04× ⚠ | — | 0.98× |
| `strict_f32*f32->f32` | 0.28× | 0.27× | 0.28× | 0.30× |
| `t_e4m3*e4m3->f32` | 1.01× | 0.96× ⚠ | 0.99× ⚠ | 1.85× |
| `t_e5m2*e5m2->f32` | — | 0.96× | — | 1.28× |

## Mixed lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `f16*e4m3->f32` | 1.05× | 1.01× | 1.05× ⚠ | 1.23× |
| `f16*e5m2->f32` | — | 1.04× ⚠ | — | 1.22× |
| `f16*fp4->f32` | 0.96× | 1.02× | 1.02× ⚠ | 1.09× |
| `f16*i4b->f32` | 1.07× | 0.98× | 0.98× | 1.25× |
| `f16*i8->f32` | 0.89× | 0.98× | 0.99× | 0.97× |
| `f16_mx_e4m3->f32` | 0.69× | 0.80× | 0.81× | 0.79× |
| `f16_mx_fp4->f32` | 0.71× | 0.82× | 0.82× | 0.82× |

## Integer lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `i8*i4b->i32` | 1.74× | 1.57× | 1.51× | **2.03×** |
| `i8*i8->i32` | 1.99× | 1.89× | 1.84× | **2.35×** |

## Machines

| Key | Device | Architecture | GPU cores | Memory | f16 baselines (GOPS, by receipt) | MLX at measurement |
|---|---|---|---:|---:|---|---|
| `m5-air` | Apple M5 | `applegpu_g17g` | 10 | 32 GB | 12,378 (integer_lanes_v3.json); 12,961 (lane_sweep_v3.json) | 0.32.3 |
| `m5-pro` | Apple M5 Pro | `applegpu_g17s` | 20 | 64 GB | 31,986 (integer_lanes_v3.json); 28,392 (lane_sweep_v3.json) | 0.32.3 |
| `m5-max` | Apple M5 Max | `applegpu_g17s` | 40 | 128 GB | 62,535 (integer_lanes_v3.json); 51,727 (lane_sweep_v3.json) | 0.32.3 |
| `m6` | Apple M6 | `applegpu_g18g` | 12 | 32 GB | 13,272 (integer_lanes_v3.json); 13,933 (lane_sweep_v3.json) | 0.32.3 |

Chip notes:

- `m5-air`: Measured on the 10-core M5 Air SKU (M5 Air ships with fewer GPU cores in some configurations); other SKUs were not measured. Cells cite the schema-3 re-run receipts (lane_sweep_v3.json, integer_lanes_v3.json; 2026-10-14); the earlier v1/v2 receipts are preserved.
- `m5-pro`: Measured on the 20-core M5 Pro SKU (18-core CPU, 64 GB); the GPU architecture string applegpu_g17s is shared with the measured M5 Max, so the SKU tier is not distinguishable from the architecture string alone. Cells cite the schema-3 full-lane re-run (lane_sweep_v3.json, which includes the e5m2 and strict/relaxed ablation lanes; integer_lanes_v3.json; 2026-10-14); the earlier receipts are preserved.
- `m5-max`: Measured on the 40-core M5 Max SKU; other M5 Max configurations were not measured. Cells cite the schema-3 re-run receipts (lane_sweep_v3.json, integer_lanes_v3.json; 2026-10-14); the v1/v2 receipts are preserved.
- `m6`: Measured on the 12-core M6 SKU, on a machine shared with other occasional users: cells cite the monitored schema-3 full-lane re-run (lane_sweep_v3.json, integer_lanes_v3.json; 2026-10-14; the operator reports that process samples during the run showed no unrelated compute). Same receipt: e4m3 runs 2.08x f16 and unscaled e5m2 1.42x f16; in this run, best-of-tiles strict/relaxed rates for f16, e4m3 and e5m2 differ by at most 1.7% and scaled e4m3 differs by 0.4% between modes, so the e4m3 advantage and the roughly half-rate scaled result persist in these aggregate measurements; per-tile rates vary. The earlier receipts are preserved.

Every cell's `vs f16` divides **its own cited receipt's** `f16*f16->f32` rate (listed per receipt above; the CSV carries each cell's baseline in `baseline_gops`/`baseline_receipt`). Same-chip normalization cancels the SKU and core count, not sequential-run clock variation between receipts.

Absolute rates are whole-GPU for the measured SKU; divide by the core count for an approximate per-core figure.

## Absolute rates (GOPS best)

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 12,453 | 32,010 | 62,587 | 13,234 |
| `e4m3*e4m3->f32` | 14,056 | 29,438 | 51,162 | 29,044 |
| `e5m2*e5m2->f32` | — | 29,652 | — | 19,785 |
| `f16*e4m3->f32` | 13,632 | 28,744 | 54,090 | 17,093 |
| `f16*e5m2->f32` | — | 29,511 | — | 16,964 |
| `f16*f16->f16` | 12,070 | 31,967 | 62,073 | 13,465 |
| `f16*f16->f32` | 12,961 | 28,392 | 51,727 | 13,933 |
| `f16*fp4->f32` | 12,483 | 28,884 | 52,938 | 15,170 |
| `f16*i4b->f32` | 13,306 | 31,476 | 61,308 | 16,531 |
| `f16*i8->f32` | 11,048 | 31,322 | 61,704 | 12,870 |
| `f16_mx_e4m3->f32` | 8,878 | 22,728 | 41,792 | 10,985 |
| `f16_mx_fp4->f32` | 9,186 | 23,342 | 42,285 | 11,420 |
| `f32*f32->f32` | 6,480 | 16,019 | 32,314 | 6,683 |
| `fp4*fp4->f32` | 12,042 | 26,300 | 48,070 | 14,964 |
| `i8*i4b->i32` | 21,494 | 50,127 | 94,360 | 26,958 |
| `i8*i8->i32` | 24,580 | 60,423 | 115,094 | 31,252 |
| `mx_e4m3*e4m3->f32` | 9,962 | 23,080 | 41,418 | 14,285 |
| `mx_e5m2*e5m2->f32` | — | 23,075 | — | 13,487 |
| `mx_fp4*fp4->f32` | 9,231 | 20,798 | 37,953 | 10,626 |
| `relaxed_mx_e4m3*e4m3->f32` | — | 23,248 | — | 14,337 |
| `strict_e4m3*e4m3->f32` | — | 29,616 | — | 28,803 |
| `strict_e5m2*e5m2->f32` | — | 29,659 | — | 19,843 |
| `strict_f16*f16->f32` | — | 29,403 | — | 13,705 |
| `strict_f32*f32->f32` | 3,646 | 7,565 | 14,526 | 4,183 |
| `t_e4m3*e4m3->f32` | 13,038 | 27,144 | 51,167 | 25,792 |
| `t_e5m2*e5m2->f32` | — | 27,302 | — | 17,824 |

## Features (observed)

| Chip | fp4_accelerated | fp8_accelerated | int8_accelerated |
|---|---|---|---|
| m5-air | no | no | yes |
| m5-pro | no | no | yes |
| m5-max | no | no | yes |
| m6 | no | yes | yes |

## Receipts

Every receipt in `receipts/` is schema-checked in CI. Completion evidence: `flag` (the receipt's own `complete`), `coverage (v3)` (requested work vs terminal rows), `coverage (inferred, v2)` — schema-2 receipts predate explicit requested-work fields, so coverage is inferred from the surviving rows and cannot prove a requested item that left none — or `legacy`/`unproven` for receipts that cannot back cells. The spread column scans every measured lane/shape in the receipt, not only cited cells.

| Receipt | Probe | Complete | Cited cells | Worst sample spread |
|---|---|---|---:|---|
| `receipts/m5-air/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 4.73× ⚠ (mx_e4m3*e4m3->f32 shared_up_fn 128x128x8) |
| `receipts/m5-air/geometry_sweep_v3.json` | `apple-matrix-capabilities/geometry_sweep` | flag | 0 | 4.77× ⚠ (mx_e4m3*e4m3->f32 hc_up 64x64x2) |
| `receipts/m5-air/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m5-air/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.62× (f16*f16->f16 [64, 64, 2] 4096x4096x4096) |
| `receipts/m5-air/integer_lanes_v3.json` | `apple-matrix-capabilities/integer_lanes` | flag | 6 | 1.66× (f16*f16->f32 [64, 64, 2] 4096x4096x4096) |
| `receipts/m5-air/lane_sweep_mixed.json` | `p23_mixed_fp8_lanes` | legacy | 0 | — |
| `receipts/m5-air/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 1.44× (f16*f16->f32 64x64x2) |
| `receipts/m5-air/lane_sweep_v3.json` | `apple-matrix-capabilities/lane_sweep` | flag | 12 | 1.45× (strict_e4m3*e4m3->f32 64x64x4) |
| `receipts/m5-air/lane_sweep_v4.json` | `p19b_mx_sweep` | legacy | 0 | — |
| `receipts/m5-max/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 4.19× ⚠ (e4m3*e4m3->f32 hc_up 64x128x4) |
| `receipts/m5-max/geometry_sweep_v3.json` | `apple-matrix-capabilities/geometry_sweep` | flag | 0 | 7.27× ⚠ (e5m2*e5m2->f32 hc_up 64x64x2) |
| `receipts/m5-max/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m5-max/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.49× (i8*i8->i32 [64, 64, 2] 2048x4096x4096) |
| `receipts/m5-max/integer_lanes_v3.json` | `apple-matrix-capabilities/integer_lanes` | flag | 6 | 1.27× (bf16*bf16->f32 [64, 64, 2] 4096x4096x4096) |
| `receipts/m5-max/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 3.09× ⚠ (f16*f16->f32 64x64x2) |
| `receipts/m5-max/lane_sweep_v3.json` | `apple-matrix-capabilities/lane_sweep` | flag | 12 | 3.04× ⚠ (strict_f16*f16->f32 64x64x2) |
| `receipts/m5-max/lane_sweep_v4.json` | `p19b_mx_sweep` | legacy | 0 | — |
| `receipts/m5-pro/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 4.26× ⚠ (mx_e4m3*e4m3->f32 hc_down 64x64x4) |
| `receipts/m5-pro/geometry_sweep_v3.json` | `apple-matrix-capabilities/geometry_sweep` | flag | 0 | 4.19× ⚠ (f16*f16->f32 shared_down_fn 64x64x4) |
| `receipts/m5-pro/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m5-pro/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.21× (i8*i8->i32 [64, 64, 2] 2048x4096x4096) |
| `receipts/m5-pro/integer_lanes_v3.json` | `apple-matrix-capabilities/integer_lanes` | flag | 6 | 1.18× (i8*i8->i32 [64, 64, 2] 2048x4096x4096) |
| `receipts/m5-pro/lane_sweep.json` | `apple-matrix-capabilities/lane_sweep` | legacy | 0 | — |
| `receipts/m5-pro/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.85× ⚠ (f16*f16->f32 64x64x4) |
| `receipts/m5-pro/lane_sweep_v2_extras.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.23× ⚠ (strict_e5m2*e5m2->f32 64x128x4) |
| `receipts/m5-pro/lane_sweep_v3.json` | `apple-matrix-capabilities/lane_sweep` | flag | 20 | 2.27× ⚠ (f16*e5m2->f32 64x64x4) |
| `receipts/m6/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 5.97× ⚠ (f16*f16->f32 deep_o_16384_m512 64x64x4) |
| `receipts/m6/geometry_sweep_v3.json` | `apple-matrix-capabilities/geometry_sweep` | flag | 0 | 5.73× ⚠ (f16*f16->f32 deep_o_16384_m512 64x64x4) |
| `receipts/m6/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m6/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.62× (f16*i4b->f32 [64, 64, 4] 4096x4096x4096) |
| `receipts/m6/integer_lanes_v2b.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.63× (f16*i4b->f32 [64, 32, 4] 4096x4096x4096) |
| `receipts/m6/integer_lanes_v2d.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.65× (f16*i4b->f32 [64, 32, 4] 4096x4096x4096) |
| `receipts/m6/integer_lanes_v3.json` | `apple-matrix-capabilities/integer_lanes` | flag | 6 | 1.47× (f16*i4b->f32 [64, 64, 2] 4096x4096x4096) |
| `receipts/m6/lane_sweep_mixed.json` | `p23_mixed_fp8_lanes` | legacy | 0 | — |
| `receipts/m6/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.40× ⚠ (e4m3*e4m3->f32 64x64x4) |
| `receipts/m6/lane_sweep_v2_extras.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.25× ⚠ (e4m3*e4m3->f32 64x64x2) |
| `receipts/m6/lane_sweep_v2b.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.10× ⚠ (t_e4m3*e4m3->f32 64x64x2) |
| `receipts/m6/lane_sweep_v2d.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 1.49× (e4m3*e4m3->f32 64x128x4) |
| `receipts/m6/lane_sweep_v3.json` | `apple-matrix-capabilities/lane_sweep` | flag | 20 | 2.05× ⚠ (e4m3*e4m3->f32 64x64x4) |
| `receipts/m6/lane_sweep_v4.json` | `p19b_mx_sweep` | legacy | 0 | — |

## Unknown chips

No measured rows yet: `m1`, `m2`, `m3`, `m4`, `m5-ultra`, `m6-pro`, `m6-max`, `a-series`.

