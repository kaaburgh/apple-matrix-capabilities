# Contributing

The most valuable contribution is a receipt from hardware we do not have. M1 through M4,
the Pro/Max/Ultra variants, and future generations are all `unknown` today.

## Add your machine

1. Run the probe from a checkout:

   ```sh
   pip install mlx numpy            # a recent MLX
   python probes/lane_sweep.py      # writes lane_sweep.json next to the script
   ```

   It needs an Apple Silicon Mac and an Xcode/SDK whose MetalPerformancePrimitives headers
   provide the tensor ops (Xcode 26.x or later; the fp8 lanes also need a macOS 27 SDK).
   Lanes that cannot build on your toolchain fail individually and are recorded as failed —
   that is a result, not an error.

2. Put the receipt at `receipts/<chip>/lane_sweep.json`. Chip keys: `m1`, `m2`, `m3`, `m4`,
   `m5-air`, `m5-max`, `m5-pro`, `m6`, ... and match the architecture id your run recorded.

3. Fill your chip's entries in `matrix/capabilities.json` from the receipt, each cell
   referring to it. Leave any cell you did not measure absent or `unknown`.

4. Open a PR. In the description, name the exact device, OS build, Xcode/SDK version and MLX
   version (they are in the receipt), and whether the machine ran on battery or AC.

## Rules

- **No cell without a receipt.** Corrections, additions and removals all need one.
- **One chip per receipt.** Do not average machines or extrapolate across generations.
- **`unknown` is a valid cell** and better than a guess. If a lane fails or is unavailable,
  record that exact status.
- **Scope is capabilities**, not implementations: no kernel code, no model benchmarks, no
  vendor speculation. If a cell is derived rather than measured, tag it `inferred` and say
  from where.
- Small PRs are welcome; there is no minimum number of rows.

## Review

Maintainers check that each receipt's device identity and check fields match the claim, and
that the protocol in METHODS.md was followed. Questions go in the PR; measured answers win.
