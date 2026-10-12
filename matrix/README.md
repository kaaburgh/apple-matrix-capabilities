# The matrix

`capabilities.json` is the machine-readable map; this page explains its fields.

- `schema`: format version (1).
- `chips.<key>`: one object per measured chip. `device_name`, `architecture` and `memory_gb`
  come from the receipt's device report; `mlx_at_measurement` names the MLX version used.
- `float_lanes`, `mixed_lanes`, `integer_lanes`: lane name → `{gops_best, vs_f16, receipt}`.
  `vs_f16` divides that chip's `f16*f16->f32` from the same receipt.
- `features`: coarse booleans as observed. `fp8_accelerated` means a measured fp8 lane
  clearly above the same-generation fp16 rate; it is not a vendor claim.
- `unknown_chips`: keys with no measured rows yet. A contributor turns one into a populated
  entry by adding a receipt and cells.

Lane naming:

- `A*B->C` — left operand format × right operand format → accumulator format;
- `t_` — transposed right operand;
- `mx_` — a UE8M0 block-scale plane on the right operand;
- `f16_mx_*` — fp16 left operand with a scaled low-bit right operand (weights-only staging
  shape).
