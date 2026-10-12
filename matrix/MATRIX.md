# Capability matrix (generated)

> Generated from [`capabilities.json`](capabilities.json) by [`tools/build_matrix.py`](../tools/build_matrix.py). Do not edit by hand; edit the JSON and run the tool.

Rates are GOPS measured at 2048×4096×4096 (`mode::multiply`), the best result the receipt records: the standard sweep tries five tile configurations and each receipt records the tiles its lanes actually ran. `vs f16` divides the same receipt's `f16*f16->f32` rate, so each chip is compared only with itself. These are lane probes, not model numbers. See [METHODS.md](../METHODS.md).

## Float lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 0.97× | 1.00× | 1.01× | — |
| `e4m3*e4m3->f32` | 1.08× | 1.01× | 1.01× | **2.08×** |
| `f16*f16->f16` | 1.04× | 1.00× | 1.00× | — |
| `f16*f16->f32` | 1.00× | 1.00× | 1.00× | 1.00× |
| `f32*f32->f32` | — | 0.53× | — | — |
| `fp4*fp4->f32` | 0.93× | 0.86× | 0.89× | 1.08× |
| `mx_e4m3*e4m3->f32` | 0.81× | 0.79× | 0.85× | 1.13× |
| `mx_fp4*fp4->f32` | 0.73× | 0.71× | 0.74× | 0.79× |
| `strict_f32*f32->f32` | — | 0.26× | — | — |
| `t_e4m3*e4m3->f32` | 1.11× | 0.93× | 1.01× | **2.17×** |

## Mixed lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `f16*e4m3->f32` | 1.05× | 1.01× | — | 1.26× |
| `f16*fp4->f32` | 0.95× | 0.98× | 1.04× | 1.08× |
| `f16*i4b->f32` | 1.11× | 0.98× | 0.99× | — |
| `f16*i8->f32` | 0.94× | 0.99× | 0.99× | — |
| `f16_mx_e4m3->f32` | 0.74× | 0.78× | — | 0.82× |
| `f16_mx_fp4->f32` | 0.73× | 0.78× | 0.83× | 0.84× |

## Integer lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `i8*i4b->i32` | 1.76× | 1.58× | 1.53× | 1.73× |
| `i8*i8->i32` | **2.03×** | 1.91× | 1.87× | 1.98× |

## Machines

| Key | Device | Architecture | GPU cores | Memory | MLX at measurement |
|---|---|---|---:|---:|---|
| `m5-air` | Apple M5 | `applegpu_g17g` | 10 | 32 GB | 0.32.3 |
| `m5-pro` | Apple M5 Pro | `applegpu_g17s` | 20 | 64 GB | 0.32.3 |
| `m5-max` | Apple M5 Max | `applegpu_g17s` | 40 | 128 GB | 0.32.3 |
| `m6` | Apple M6 | `applegpu_g18g` | 12 | 32 GB | 0.32.3 |

Absolute rates are whole-GPU for the measured SKU; divide by the core count for an approximate per-core figure. The `vs f16` ratios are same-chip and cancel the core count.

## Absolute rates (GOPS best)

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 11,728 | 31,875 | 62,520 | — |
| `e4m3*e4m3->f32` | 14,062 | 29,435 | 54,590 | 28,800 |
| `f16*e4m3->f32` | 13,743 | 29,428 | — | 17,687 |
| `f16*f16->f16` | 12,563 | 31,898 | 62,057 | — |
| `f16*f16->f32` | 13,040 | 29,256 | 53,885 | 13,858 |
| `f16*fp4->f32` | 12,511 | 28,542 | 55,955 | 15,053 |
| `f16*i4b->f32` | 13,354 | 31,300 | 61,508 | — |
| `f16*i8->f32` | 11,260 | 31,473 | 61,644 | — |
| `f16_mx_e4m3->f32` | 9,743 | 22,683 | — | 11,528 |
| `f16_mx_fp4->f32` | 9,528 | 22,899 | 44,765 | 11,687 |
| `f32*f32->f32` | — | 15,392 | — | — |
| `fp4*fp4->f32` | 12,082 | 25,152 | 47,734 | 14,919 |
| `i8*i4b->i32` | 21,151 | 50,230 | 95,015 | 23,527 |
| `i8*i8->i32` | 24,471 | 60,901 | 116,369 | 26,925 |
| `mx_e4m3*e4m3->f32` | 10,517 | 23,097 | 45,950 | 15,610 |
| `mx_fp4*fp4->f32` | 9,506 | 20,851 | 39,990 | 11,005 |
| `strict_f32*f32->f32` | — | 7,572 | — | — |
| `t_e4m3*e4m3->f32` | 14,523 | 27,114 | 54,345 | 30,006 |

## Features (observed)

| Chip | fp4_accelerated | fp8_accelerated | int8_accelerated |
|---|---|---|---|
| m5-air | no | no | yes |
| m5-pro | no | no | yes |
| m5-max | no | no | yes |
| m6 | no | yes | yes |

## Unknown chips

No measured rows yet: `m1`, `m2`, `m3`, `m4`, `m5-ultra`, `m6-pro`, `m6-max`, `a-series`.

