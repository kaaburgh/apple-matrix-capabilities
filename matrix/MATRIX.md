# Capability matrix (generated)

> Generated from [`capabilities.json`](capabilities.json) by [`tools/build_matrix.py`](../tools/build_matrix.py). Do not edit by hand; edit the JSON and run the tool.

Rates are GOPS measured at 2048×4096×4096 (`mode::multiply`), the best result the receipt records: the standard sweep tries five tile configurations and each receipt records the tiles its lanes actually ran. `vs f16` divides the same receipt's `f16*f16->f32` rate, so each chip is compared only with itself. These are lane probes, not model numbers. See [METHODS.md](../METHODS.md).

## Float lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `bf16*bf16->f32` | 1.00× | 1.00× | 1.00× | 0.98× |
| `e4m3*e4m3->f32` | 1.06× | 1.01× | 1.02× | **2.07×** |
| `e5m2*e5m2->f32` | — | 1.01× | — | 1.44× |
| `f16*f16->f16` | 1.00× | 1.00× | 1.01× | 1.07× |
| `f16*f16->f32` | 1.00× | 1.00× | 1.00× | 1.00× |
| `f32*f32->f32` | 0.50× | 0.53× | 0.62× | 0.51× |
| `fp4*fp4->f32` | 0.91× | 0.86× | 0.93× | 1.08× |
| `mx_e4m3*e4m3->f32` | 0.74× | 0.79× | 0.82× | 1.04× |
| `mx_e5m2*e5m2->f32` | — | 0.80× | — | 0.99× |
| `mx_fp4*fp4->f32` | 0.68× | 0.71× | 0.74× | 0.77× |
| `relaxed_mx_e4m3*e4m3->f32` | — | 0.71× | — | 1.05× |
| `strict_e4m3*e4m3->f32` | — | 1.01× | — | **2.12×** |
| `strict_e5m2*e5m2->f32` | — | 1.03× | — | 1.45× |
| `strict_f16*f16->f32` | — | 0.99× | — | 1.01× |
| `strict_f32*f32->f32` | 0.28× | 0.26× | 0.28× | 0.30× |
| `t_e4m3*e4m3->f32` | 0.98× | 0.93× | 1.00× | 1.88× |
| `t_e5m2*e5m2->f32` | — | 0.93× | — | 1.31× |

## Mixed lanes

| Lane | M5 Air (10c) | M5 Pro (20c) | M5 Max (40c) | M6 (12c) |
|---|--:|--:|--:|--:|
| `f16*e4m3->f32` | 1.03× | 1.01× | 1.08× | 1.24× |
| `f16*e5m2->f32` | — | 1.01× | — | 1.25× |
| `f16*fp4->f32` | 0.94× | 0.98× | 1.00× | 1.10× |
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

## Unknown chips

No measured rows yet: `m1`, `m2`, `m3`, `m4`, `m5-ultra`, `m6-pro`, `m6-max`, `a-series`.

