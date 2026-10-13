# The matrix

`capabilities.json` is the machine-readable map and the single source of truth. The views
[`MATRIX.md`](MATRIX.md) and [`capabilities.csv`](capabilities.csv) are generated from it by
[`tools/build_matrix.py`](../tools/build_matrix.py) — do not edit them by hand. A JSON Schema
for contributors is in [`schema.json`](schema.json); the build tool performs the stricter
checks a schema cannot: receipt path containment, lane existence and status, measured-rate
fidelity, completion evidence, a successful same-receipt `f16` baseline, symlink-safe path
containment, and device identity (device name, architecture, memory, MLX version — and the
receipt's own `gpu_cores` once a receipt carries it).

## Fields

- `schema`: format version (1).
- `chips.<key>`: one object per measured chip. `device_name`, `architecture` and `memory_gb`
  come from the receipt's device report; `gpu_cores` is the measured SKU's GPU core count;
  `note` records SKU specifics; `mlx_at_measurement` names the MLX version used.
- Absolute rates are whole-GPU for the measured SKU; the generated CSV also carries
  `gops_per_core`. `vs_f16` is same-chip, so core count cancels.
- `float_lanes`, `mixed_lanes`, `integer_lanes`: lane name → `{gops_best, vs_f16, receipt}`.
  `vs_f16` divides that chip's `f16*f16->f32` from the same receipt.
- `features`: coarse booleans as observed. `fp8_accelerated` means a measured fp8 lane
  clearly above the same-generation fp16 rate; it is not a vendor claim.
- `unknown_chips`: keys with no measured rows yet. A contributor turns one into a populated
  entry by adding a receipt and cells.

Lane naming:

- `A*B->C` — left operand format × right operand format → declared destination type
  (not a claim about internal accumulator precision; see METHODS.md);
- `t_` — transposed right operand;
- `mx_` — a UE8M0 block-scale plane on the right operand;
- `strict_` — the lane runs with `relaxed_precision` off (scaled lanes are strict by default);
- `relaxed_` — a scaled lane runs with `relaxed_precision` on;
- `f16_mx_*` — fp16 left operand with a scaled low-bit right operand (weights-only staging
  shape).
