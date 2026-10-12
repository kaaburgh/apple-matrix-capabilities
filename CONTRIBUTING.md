# Contributing

The most valuable contribution is a receipt from hardware we do not have. M1 through M4,
M5 Ultra, the M6 Pro/Max variants and future generations are `unknown` today.

## Add your machine

1. Run the probes from a checkout:

   ```sh
   python3 -m venv .venv && source .venv/bin/activate   # or a manager of your choice
   pip install mlx numpy                    # tested with Python 3.12, MLX 0.32.3, NumPy 2.x
   python probes/lane_sweep.py              # writes lane_sweep.json next to the script
   python probes/integer_lanes.py --json integer_lanes.json
   ```

   Run it on a quiet, thermally settled machine; one benchmark at a time; note AC/battery and
   power mode (see "Run conditions" in METHODS.md).

   It needs an Apple Silicon Mac and an Xcode/SDK whose MetalPerformancePrimitives headers
   provide the tensor ops. Currently that means the **Xcode 27 SDK**: with Xcode 26.6,
   MLX (0.29-0.32) cannot JIT-compile even fp16 lanes, because its injected kernel headers
   reference fp8 `metal` types the older Metal standard library lacks; pinning an older MLX
   does not help (the probe uses `mx.device_info`). When lanes do compile, a lane may still
   fail its checks or build individually — that too is a result, recorded as such.

2. Put the receipts at `receipts/<chip>/lane_sweep.json` and
   `receipts/<chip>/integer_lanes.json`. Chip keys: `m1`, `m2`, `m3`, `m4`,
   `m5-air`, `m5-max`, `m5-pro`, `m6`, ... and match the architecture id your run recorded.

3. Fill your chip's entries in `matrix/capabilities.json` from the receipts, each cell
   referring to one. Leave any cell you did not measure absent or `unknown`.

4. Regenerate the views and commit them: `pip install jsonschema && python
   tools/build_matrix.py`. CI (on PRs and on `main`) validates against `matrix/schema.json`,
   checks every cell against the receipt it cites (lane existence and status, measured rate,
   baseline, device identity), runs the validator's regression tests, and fails if the views
   are stale. Edit `capabilities.json` only — never the generated files.

5. Open a PR. In the description, name the exact device, OS build, Xcode/SDK version and MLX
   version (they are in the receipt), and whether the machine ran on battery or AC.

## Rules

- **No cell without a receipt.** Corrections, additions and removals all need one.
- **One chip per receipt.** Do not average machines or extrapolate across generations.
- **`unknown` is a valid cell** and better than a guess. If a lane fails or is unavailable,
  record that exact status.
- **Scope is capabilities**, not implementations: no model benchmarks and no vendor
  speculation; the probe kernels in `probes/` are the point. If a cell is derived rather than
  measured, tag it `inferred` and say from where.
- Small PRs are welcome; there is no minimum number of rows.

## Review

Maintainers check that each receipt's device identity and check fields match the claim, and
that the protocol in METHODS.md was followed. Questions go in the PR; measured answers win.
