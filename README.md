# Apple matrix capabilities

A per-generation, evidence-first map of what Apple Silicon GPUs can do with low-bit matrix
operations: which operand formats the matrix units consume, at what measured rates, and what
restricts each one — with the probes that produced every number.

There is no vendor table for this. Apple documents neither the operand-format menu of the
tensor units nor its per-generation rates; the closest public truths are the
MetalPerformancePrimitives headers and the source of projects like MLX, both version-bound.
Every project ends up measuring the same things separately. This repository collects the
measurements in one place, one receipt per cell.

**Start here:** read the map in [`matrix/MATRIX.md`](matrix/MATRIX.md) — or the raw grid in
[`matrix/capabilities.csv`](matrix/capabilities.csv). Both are generated from
[`matrix/capabilities.json`](matrix/capabilities.json).

## What is here

| Path | |
|---|---|
| `matrix/capabilities.json` | The map: per chip, measured lanes (rate and vs-fp16), features, and the receipt behind each cell |
| `matrix/MATRIX.md`, `matrix/capabilities.csv` | Generated views (do not edit by hand; run `tools/build_matrix.py`) |
| `probes/lane_sweep.py`, `probes/integer_lanes.py`, `probes/geometry_sweep.py` | The probes that produce receipts: operand-format sweeps through `mpp::tensor_ops::matmul2d` (lane/format table, integer lanes, and the optional geometry dimension) |
| `tools/build_matrix.py` | Validator and view generator; `--check` runs in CI |
| `tests/` | Validator regression tests (`python -m unittest discover -s tests`) |
| `receipts/<chip>/` | The receipts behind every observed cell |
| `METHODS.md` | Protocol, traps, and what the numbers do and do not mean |
| `CONTRIBUTING.md` | How to add your machine's rows |

## How to read it

- **Evidence tags**: `observed` (a receipt in `receipts/`), `reported` (a third party's
  measurement, linked), `inferred` (reasoned from source or headers, no local run),
  `unknown` (no claim).
- **Rates**: GOPS at a fixed shape (2048×4096×4096, `mode::multiply`), the best result the
   receipt records — the standard sweep tries five tile configurations, and each receipt lists
   the tiles its lanes actually ran. `vs_f16` divides by the same receipt's `f16*f16->f32`, so
   each chip is compared only with itself.
- **These are lane probes, not model numbers.** A 2× lane does not mean a 2× model.

## Coverage today

Seed rows measured by Precisit, October 2026: M5 Air (`applegpu_g17g`), M5 Pro and M5 Max
(both `applegpu_g17s` — the architecture string does not separate the tiers; core count and
memory do), M6 (`applegpu_g18g`). Everything else is `unknown` until someone runs the probes
— that is the main way to contribute.

## Related projects

- [`philipturner/metal-benchmarks`](https://github.com/philipturner/metal-benchmarks) — Apple
  GPU microarchitecture benchmarks (M1/M2 era).
- [`Anemll/fp8-mlp-metal41-bench`](https://github.com/Anemll/fp8-mlp-metal41-bench) — FP8 MLP
  benchmarking on Metal 4.1.
- [`pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8`](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8)
  — matmul2d dtype probes in a ComfyUI extension.
- [`v-code01/silent-fp16`](https://github.com/v-code01/silent-fp16) — M4 fp16-accumulator
  precision finding.
- [`tsotchke/tensorcore`](https://github.com/tsotchke/tensorcore) — kernel library wrapping
  `simdgroup_matrix` and `mpp::tensor_ops`.

## License

Scripts: MIT. Data and text: CC BY 4.0. See [LICENSES.md](LICENSES.md).
