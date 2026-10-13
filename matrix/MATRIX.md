# Capability matrix (generated)

> Generated from [`capabilities.json`](capabilities.json) by [`tools/build_matrix.py`](../tools/build_matrix.py). Do not edit by hand; edit the JSON and run the tool.

Rates are GOPS measured at 2048×4096×4096 (`mode::multiply`), the best result the receipt records: the standard sweep tries five tile configurations and each receipt records the tiles its lanes actually ran. `vs f16` divides the same receipt's `f16*f16->f32` rate, so each chip is compared only with itself. A `⚠` marks a cell whose cited receipt shows a max/min timing spread above 2× across samples: treat it as unstable and rerun before building any dispatch rule on it. These are lane probes, not model numbers. See [METHODS.md](../METHODS.md) and [README.md](../README.md) for the evidence-status model (only observed, receipt-backed rows are published).

## Float lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 1.00× | 1.00× | 1.00× | 0.98× |
| `e4m3*e4m3->f32` | 1.06× | 1.01× | 1.02× ⚠ | **2.07×** |
| `e5m2*e5m2->f32` | — | 1.01× | — | 1.44× |
| `f16*f16->f16` | 1.00× | 1.00× | 1.01× | 1.07× |
| `f16*f16->f32` | 1.00× | 1.00× ⚠ | 1.00× ⚠ | 1.00× |
| `f32*f32->f32` | 0.50× | 0.53× | 0.62× | 0.51× |
| `fp4*fp4->f32` | 0.91× | 0.86× | 0.93× ⚠ | 1.08× |
| `mx_e4m3*e4m3->f32` | 0.74× | 0.79× | 0.82× | 1.04× |
| `mx_e5m2*e5m2->f32` | — | 0.80× | — | 0.99× |
| `mx_fp4*fp4->f32` | 0.68× | 0.71× | 0.74× | 0.77× |
| `relaxed_mx_e4m3*e4m3->f32` | — | 0.71× | — | 1.05× |
| `strict_e4m3*e4m3->f32` | — | 1.01× ⚠ | — | **2.12×** |
| `strict_e5m2*e5m2->f32` | — | 1.03× ⚠ | — | 1.45× |
| `strict_f16*f16->f32` | — | 0.99× ⚠ | — | 1.01× |
| `strict_f32*f32->f32` | 0.28× | 0.26× | 0.28× | 0.30× |
| `t_e4m3*e4m3->f32` | 0.98× | 0.93× | 1.00× ⚠ | 1.88× |
| `t_e5m2*e5m2->f32` | — | 0.93× | — | 1.31× |

## Mixed lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `f16*e4m3->f32` | 1.03× | 1.01× ⚠ | 1.08× ⚠ | 1.24× |
| `f16*e5m2->f32` | — | 1.01× ⚠ | — | 1.25× |
| `f16*fp4->f32` | 0.94× | 0.98× ⚠ | 1.00× | 1.10× |
| `f16*i4b->f32` | 1.10× | 0.98× | 0.99× | 1.24× |
| `f16*i8->f32` | 0.90× | 0.99× | 0.98× | 0.93× |
| `f16_mx_e4m3->f32` | 0.67× | 0.78× | 0.81× | 0.80× |
| `f16_mx_fp4->f32` | 0.68× | 0.78× | 0.84× | 0.84× |

## Integer lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `i8*i4b->i32` | 1.77× | 1.58× | 1.51× | **2.03×** |
| `i8*i8->i32` | **2.12×** | 1.91× | 1.85× | **2.31×** |

## Machines

| Key | Device | Architecture | GPU cores | Memory | f16 baselines (GOPS, by receipt) | MLX at measurement |
|---|---|---|---:|---:|---|---|
| `m5-air` | Apple M5 | `applegpu_g17g` | 10 | 32 GB | 12,185 (integer_lanes_v2.json); 13,309 (lane_sweep_v2.json) | 0.32.3 |
| `m5-pro` | Apple M5 Pro | `applegpu_g17s` | 20 | 64 GB | 31,869 (integer_lanes_v2.json); 29,256 (lane_sweep_v2.json); 29,138 (lane_sweep_v2_extras.json) | 0.32.3 |
| `m5-max` | Apple M5 Max | `applegpu_g17s` | 40 | 128 GB | 62,030 (integer_lanes_v2.json); 51,721 (lane_sweep_v2.json) | 0.32.3 |
| `m6` | Apple M6 | `applegpu_g18g` | 12 | 32 GB | 13,270 (integer_lanes_v2d.json); 13,655 (lane_sweep_v2_extras.json); 13,795 (lane_sweep_v2d.json) | 0.32.3 |

Chip notes:

- `m5-air`: Measured on the 10-core M5 Air SKU (M5 Air ships with fewer GPU cores in some configurations); other SKUs were not measured. Cells cite the schema-2 receipts with decoded-reference checks (lane_sweep_v2.json, integer_lanes_v2.json); the earlier v1 receipts are preserved.
- `m5-pro`: Measured on the 20-core M5 Pro SKU (18-core CPU, 64 GB); the GPU architecture string applegpu_g17s is shared with the measured M5 Max, so the SKU tier is not distinguishable from the architecture string alone. Cells cite the schema-2 receipts with decoded-reference checks (lane_sweep_v2.json, integer_lanes_v2.json); the earlier v1 receipts are preserved. The e5m2 and strict/relaxed ablation lanes cite lane_sweep_v2_extras.json.
- `m5-max`: Measured on the 40-core M5 Max SKU; other M5 Max configurations were not measured. Cells cite the schema-2 receipts with decoded-reference checks (lane_sweep_v2.json, integer_lanes_v2.json); the earlier v1 receipts are preserved.
- `m6`: Measured on the 12-core M6 SKU, on a machine shared with other occasional users: cells cite the monitored schema-2 runs (lane_sweep_v2d.json, lane_sweep_v2_extras.json, integer_lanes_v2d.json; process samples taken during the windows, all ours). Ablations in extras: strict == relaxed for f16/e4m3/e5m2 and scaled e4m3 is identical relaxed vs strict, so the e4m3 ~2x and the half-rate scale-plane path are precision-mode-independent; unscaled e5m2 runs ~1.44x f16, scaled ~1.0x. Float lanes agree within ~1% across runs; integer lanes varied up to ~9% (i8*i8 28.1-30.6k). Earlier receipts are preserved; the transposed fp8 lanes read 8-13% higher in the pre-v2 receipt, unresolved.

Every cell's `vs f16` divides **its own cited receipt's** `f16*f16->f32` rate (listed per receipt above; the CSV carries each cell's baseline in `baseline_gops`/`baseline_receipt`). Same-chip normalization cancels the SKU and core count, not sequential-run clock variation between receipts.

Absolute rates are whole-GPU for the measured SKU; divide by the core count for an approximate per-core figure.

## Absolute rates (GOPS best)

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 12,213 | 31,875 | 62,103 | 13,038 |
| `e4m3*e4m3->f32` | 14,063 | 29,435 | 52,977 | 28,602 |
| `e5m2*e5m2->f32` | — | 29,476 | — | 19,704 |
| `f16*e4m3->f32` | 13,709 | 29,428 | 56,048 | 17,046 |
| `f16*e5m2->f32` | — | 29,503 | — | 17,047 |
| `f16*f16->f16` | 12,176 | 31,898 | 62,453 | 14,176 |
| `f16*f16->f32` | 13,309 | 29,256 | 51,721 | 13,795 |
| `f16*fp4->f32` | 12,452 | 28,542 | 51,921 | 15,198 |
| `f16*i4b->f32` | 13,428 | 31,300 | 61,535 | 16,481 |
| `f16*i8->f32` | 10,956 | 31,473 | 60,974 | 12,322 |
| `f16_mx_e4m3->f32` | 8,885 | 22,683 | 41,722 | 10,978 |
| `f16_mx_fp4->f32` | 9,074 | 22,899 | 43,311 | 11,589 |
| `f32*f32->f32` | 6,683 | 15,392 | 32,245 | 6,980 |
| `fp4*fp4->f32` | 12,096 | 25,152 | 48,193 | 14,947 |
| `i8*i4b->i32` | 21,611 | 50,230 | 93,691 | 26,889 |
| `i8*i8->i32` | 25,809 | 60,901 | 114,818 | 30,639 |
| `mx_e4m3*e4m3->f32` | 9,827 | 23,097 | 42,496 | 14,364 |
| `mx_e5m2*e5m2->f32` | — | 23,172 | — | 13,494 |
| `mx_fp4*fp4->f32` | 9,101 | 20,851 | 38,516 | 10,627 |
| `relaxed_mx_e4m3*e4m3->f32` | — | 20,843 | — | 14,354 |
| `strict_e4m3*e4m3->f32` | — | 29,374 | — | 28,978 |
| `strict_e5m2*e5m2->f32` | — | 29,887 | — | 19,733 |
| `strict_f16*f16->f32` | — | 28,980 | — | 13,838 |
| `strict_f32*f32->f32` | 3,672 | 7,572 | 14,523 | 4,201 |
| `t_e4m3*e4m3->f32` | 13,045 | 27,114 | 51,622 | 25,860 |
| `t_e5m2*e5m2->f32` | — | 26,981 | — | 17,839 |

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
| `receipts/m5-air/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m5-air/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 6 | 1.62× (f16*f16->f16 [64, 64, 2] 4096x4096x4096) |
| `receipts/m5-air/lane_sweep_mixed.json` | `p23_mixed_fp8_lanes` | legacy | 0 | — |
| `receipts/m5-air/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 12 | 1.44× (f16*f16->f32 64x64x2) |
| `receipts/m5-air/lane_sweep_v4.json` | `p19b_mx_sweep` | legacy | 0 | — |
| `receipts/m5-max/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 4.19× ⚠ (e4m3*e4m3->f32 hc_up 64x128x4) |
| `receipts/m5-max/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m5-max/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 6 | 1.49× (i8*i8->i32 [64, 64, 2] 2048x4096x4096) |
| `receipts/m5-max/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 12 | 3.09× ⚠ (f16*f16->f32 64x64x2) |
| `receipts/m5-max/lane_sweep_v4.json` | `p19b_mx_sweep` | legacy | 0 | — |
| `receipts/m5-pro/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 4.26× ⚠ (mx_e4m3*e4m3->f32 hc_down 64x64x4) |
| `receipts/m5-pro/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m5-pro/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 6 | 1.21× (i8*i8->i32 [64, 64, 2] 2048x4096x4096) |
| `receipts/m5-pro/lane_sweep.json` | `apple-matrix-capabilities/lane_sweep` | legacy | 0 | — |
| `receipts/m5-pro/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 12 | 2.85× ⚠ (f16*f16->f32 64x64x4) |
| `receipts/m5-pro/lane_sweep_v2_extras.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 8 | 2.23× ⚠ (strict_e5m2*e5m2->f32 64x128x4) |
| `receipts/m6/geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep` | coverage (inferred, v2) | 0 | 5.97× ⚠ (f16*f16->f32 deep_o_16384_m512 64x64x4) |
| `receipts/m6/integer_lanes.json` | `legacy (pre-schema)` | legacy | 0 | — |
| `receipts/m6/integer_lanes_v2.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.62× (f16*i4b->f32 [64, 64, 4] 4096x4096x4096) |
| `receipts/m6/integer_lanes_v2b.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 0 | 1.63× (f16*i4b->f32 [64, 32, 4] 4096x4096x4096) |
| `receipts/m6/integer_lanes_v2d.json` | `apple-matrix-capabilities/integer_lanes` | coverage (inferred, v2) | 6 | 1.65× (f16*i4b->f32 [64, 32, 4] 4096x4096x4096) |
| `receipts/m6/lane_sweep_mixed.json` | `p23_mixed_fp8_lanes` | legacy | 0 | — |
| `receipts/m6/lane_sweep_v2.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.40× ⚠ (e4m3*e4m3->f32 64x64x4) |
| `receipts/m6/lane_sweep_v2_extras.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 8 | 2.25× ⚠ (e4m3*e4m3->f32 64x64x2) |
| `receipts/m6/lane_sweep_v2b.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 0 | 2.10× ⚠ (t_e4m3*e4m3->f32 64x64x2) |
| `receipts/m6/lane_sweep_v2d.json` | `apple-matrix-capabilities/lane_sweep` | coverage (inferred, v2) | 12 | 1.49× (e4m3*e4m3->f32 64x128x4) |
| `receipts/m6/lane_sweep_v4.json` | `p19b_mx_sweep` | legacy | 0 | — |

## Unknown chips

No measured rows yet: `m1`, `m2`, `m3`, `m4`, `m5-ultra`, `m6-pro`, `m6-max`, `a-series`.

