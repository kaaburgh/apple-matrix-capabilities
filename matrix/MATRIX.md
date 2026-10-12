# Capability matrix (generated)

> Generated from [`capabilities.json`](capabilities.json) by [`tools/build_matrix.py`](../tools/build_matrix.py). Do not edit by hand; edit the JSON and run the tool.

Rates are GOPS measured at 2048×4096×4096 (best of five tiles, `mode::multiply`); `vs f16` divides the same receipt's `f16*f16` rate, so each chip is compared only with itself. These are lane probes, not model numbers. See [METHODS.md](../METHODS.md).

## Float lanes

| Lane | M5 Air | M5 Max | M6 |
|---|--:|--:|--:|
| `bf16*bf16->f32` | 0.97× | 1.01× | — |
| `e4m3*e4m3->f32` | 1.08× | 1.01× | **2.08×** |
| `f16*f16->f16` | 1.04× | 1.00× | — |
| `f16*f16->f32` | 1.00× | 1.00× | 1.00× |
| `fp4*fp4->f32` | 0.93× | 0.89× | 1.08× |
| `mx_e4m3*e4m3->f32` | 0.81× | 0.85× | 1.13× |
| `mx_fp4*fp4->f32` | 0.73× | 0.74× | 0.79× |
| `t_e4m3*e4m3->f32` | 1.11× | 1.01× | **2.17×** |

## Mixed lanes

| Lane | M5 Air | M5 Max | M6 |
|---|--:|--:|--:|
| `f16*e4m3->f32` | 1.05× | — | 1.26× |
| `f16*fp4->f32` | 0.95× | 1.04× | 1.08× |
| `f16*i4b->f32` | 1.11× | 0.99× | — |
| `f16*i8->f32` | 0.94× | 0.99× | — |
| `f16_mx_e4m3->f32` | 0.74× | — | 0.82× |
| `f16_mx_fp4->f32` | 0.73× | 0.83× | 0.84× |

## Integer lanes

| Lane | M5 Air | M5 Max | M6 |
|---|--:|--:|--:|
| `i8*i4b->i32` | 1.76× | 1.53× | 1.73× |
| `i8*i8->i32` | **2.03×** | 1.87× | 1.98× |

## Absolute rates (GOPS best)

| Lane | M5 Air | M5 Max | M6 |
|---|--:|--:|--:|
| `bf16*bf16->f32` | 11,728 | 62,520 | — |
| `e4m3*e4m3->f32` | 14,062 | 54,590 | 28,800 |
| `f16*e4m3->f32` | 13,743 | — | 17,687 |
| `f16*f16->f16` | 12,563 | 62,057 | — |
| `f16*f16->f32` | 13,040 | 53,885 | 13,858 |
| `f16*fp4->f32` | 12,511 | 55,955 | 15,053 |
| `f16*i4b->f32` | 13,354 | 61,508 | — |
| `f16*i8->f32` | 11,260 | 61,644 | — |
| `f16_mx_e4m3->f32` | 9,743 | — | 11,528 |
| `f16_mx_fp4->f32` | 9,528 | 44,765 | 11,687 |
| `fp4*fp4->f32` | 12,082 | 47,734 | 14,919 |
| `i8*i4b->i32` | 21,151 | 95,015 | 23,527 |
| `i8*i8->i32` | 24,471 | 116,369 | 26,925 |
| `mx_e4m3*e4m3->f32` | 10,517 | 45,950 | 15,610 |
| `mx_fp4*fp4->f32` | 9,506 | 39,990 | 11,005 |
| `t_e4m3*e4m3->f32` | 14,523 | 54,345 | 30,006 |

## Features (observed)

| Chip | fp4_accelerated | fp8_accelerated | int8_accelerated |
|---|---|---|---|
| m5-air | no | no | yes |
| m5-max | no | no | yes |
| m6 | no | yes | yes |

## Unknown chips

No measured rows yet: `m1`, `m2`, `m3`, `m4`, `m5-pro`, `m5-ultra`, `m6-pro`, `m6-max`, `a-series`.

