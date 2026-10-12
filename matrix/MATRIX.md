# Capability matrix (generated)

> Generated from [`capabilities.json`](capabilities.json) by [`tools/build_matrix.py`](../tools/build_matrix.py). Do not edit by hand; edit the JSON and run the tool.

Rates are GOPS measured at 2048×4096×4096 (`mode::multiply`), the best result the receipt records: the standard sweep tries five tile configurations and each receipt records the tiles its lanes actually ran. `vs f16` divides the same receipt's `f16*f16->f32` rate, so each chip is compared only with itself. These are lane probes, not model numbers. See [METHODS.md](../METHODS.md).

## Float lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 1.00× | 1.00× | 1.01× | 0.98× |
| `e4m3*e4m3->f32` | 1.06× | 1.01× | 1.01× | **2.07×** |
| `f16*f16->f16` | 1.00× | 1.00× | 1.00× | 1.07× |
| `f16*f16->f32` | 1.00× | 1.00× | 1.00× | 1.00× |
| `f32*f32->f32` | 0.50× | 0.53× | — | 0.51× |
| `fp4*fp4->f32` | 0.91× | 0.86× | 0.89× | 1.08× |
| `mx_e4m3*e4m3->f32` | 0.74× | 0.79× | 0.85× | 1.04× |
| `mx_fp4*fp4->f32` | 0.68× | 0.71× | 0.74× | 0.77× |
| `strict_f32*f32->f32` | 0.28× | 0.26× | — | 0.30× |
| `t_e4m3*e4m3->f32` | 0.98× | 0.93× | 1.01× | 1.88× |

## Mixed lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `f16*e4m3->f32` | 1.03× | 1.01× | — | 1.24× |
| `f16*fp4->f32` | 0.94× | 0.98× | 1.04× | 1.10× |
| `f16*i4b->f32` | 1.10× | 0.98× | 0.99× | 1.24× |
| `f16*i8->f32` | 0.90× | 0.99× | 0.99× | 0.93× |
| `f16_mx_e4m3->f32` | 0.67× | 0.78× | — | 0.80× |
| `f16_mx_fp4->f32` | 0.68× | 0.78× | 0.83× | 0.84× |

## Integer lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `i8*i4b->i32` | 1.77× | 1.58× | 1.53× | **2.03×** |
| `i8*i8->i32` | **2.12×** | 1.91× | 1.87× | **2.31×** |

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
| `bf16*bf16->f32` | 12,213 | 31,875 | 62,520 | 13,038 |
| `e4m3*e4m3->f32` | 14,063 | 29,435 | 54,590 | 28,602 |
| `f16*e4m3->f32` | 13,709 | 29,428 | — | 17,046 |
| `f16*f16->f16` | 12,176 | 31,898 | 62,057 | 14,176 |
| `f16*f16->f32` | 13,309 | 29,256 | 53,885 | 13,795 |
| `f16*fp4->f32` | 12,452 | 28,542 | 55,955 | 15,198 |
| `f16*i4b->f32` | 13,428 | 31,300 | 61,508 | 16,481 |
| `f16*i8->f32` | 10,956 | 31,473 | 61,644 | 12,322 |
| `f16_mx_e4m3->f32` | 8,885 | 22,683 | — | 10,978 |
| `f16_mx_fp4->f32` | 9,074 | 22,899 | 44,765 | 11,589 |
| `f32*f32->f32` | 6,683 | 15,392 | — | 6,980 |
| `fp4*fp4->f32` | 12,096 | 25,152 | 47,734 | 14,947 |
| `i8*i4b->i32` | 21,611 | 50,230 | 95,015 | 26,889 |
| `i8*i8->i32` | 25,809 | 60,901 | 116,369 | 30,639 |
| `mx_e4m3*e4m3->f32` | 9,827 | 23,097 | 45,950 | 14,364 |
| `mx_fp4*fp4->f32` | 9,101 | 20,851 | 39,990 | 10,627 |
| `strict_f32*f32->f32` | 3,672 | 7,572 | — | 4,201 |
| `t_e4m3*e4m3->f32` | 13,045 | 27,114 | 54,345 | 25,860 |

## Features (observed)

| Chip | fp4_accelerated | fp8_accelerated | int8_accelerated |
|---|---|---|---|
| m5-air | no | no | yes |
| m5-pro | no | no | yes |
| m5-max | no | no | yes |
| m6 | no | yes | yes |

## Unknown chips

No measured rows yet: `m1`, `m2`, `m3`, `m4`, `m5-ultra`, `m6-pro`, `m6-max`, `a-series`.

