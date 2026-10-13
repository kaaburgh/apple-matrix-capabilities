# Contributing

The most valuable contribution is a receipt from hardware we do not have. M1 through M4,
M5 Ultra, the M6 Pro/Max variants and future generations are `unknown` today.

## Add your machine

1. Run the probes from a checkout:

   ```sh
   python3 -m venv .venv && source .venv/bin/activate   # or a manager of your choice
   pip install mlx numpy                    # tested with Python 3.12, MLX 0.32.3, NumPy 2.x
   python probes/lane_sweep.py              # writes lane_sweep.json in the working directory
   python probes/integer_lanes.py --json integer_lanes.json
   ```

   Both probes refuse to overwrite an existing receipt unless `--force` is passed; prefer a
   new name. Receipts are written atomically and end with `complete: true`; never hand-edit a
   receipt (corrections are new receipts, and an interrupted run's `complete: false` stays as
   it is). Run on a quiet, thermally settled machine; one benchmark at a time; note
   AC/battery and power mode (see "Run conditions" in METHODS.md).

   It needs an Apple Silicon Mac and an Xcode/SDK whose MetalPerformancePrimitives headers
   provide the tensor ops. Currently that means the **Xcode 27 SDK**: with Xcode 26.6,
   MLX (0.29-0.32) cannot JIT-compile even fp16 lanes, because its injected kernel headers
   reference fp8 `metal` types the older Metal standard library lacks; pinning an older MLX
   does not help (the probe uses `mx.device_info`). When lanes do compile, a lane may still
   fail its checks or build individually — that too is a result, recorded as such.

2. Put the receipts at `receipts/<chip>/lane_sweep.json` and
   `receipts/<chip>/integer_lanes.json`. Chip keys: `m1`, `m2`, `m3`, `m4`,
   `m5-air`, `m5-max`, `m5-pro`, `m6`, ... and match the architecture id your run recorded.
   Re-runs: keep every receipt; give a repeat run a distinct name (for example
   `lane_sweep_2.json`) and cite that file from the cell — never overwrite evidence.

3. Fill your chip's entries in `matrix/capabilities.json` from the receipts, each cell
   referring to one. The matrix publishes **observed** rows only: leave any cell you did not
   measure absent (unmeasured chips are listed in `unknown_chips`), and keep failed or partial
   runs in the receipt where the status fields show exactly what happened. Evidence tags other
   than `observed` (`reported`, `inferred`) are reserved and cannot carry measured cells
   today.

4. Regenerate the views and commit them:
   `pip install 'jsonschema>=4.18,<5' && python tools/build_matrix.py`. CI (on PRs and on
   `main`, Python 3.12) validates `matrix/capabilities.json` against `matrix/schema.json`,
   validates **every** receipt under `receipts/` against `receipts/schema.json` and the shared
   semantic rules (strict JSON with no non-finite numbers anywhere, statuses, check/error
   consistency, rate-vs-samples and gops bindings, protocol shape/mode, completion evidence),
   checks every cell against the receipt it cites (lane existence and status, measured rate,
   per-cell baseline, device identity, symlink-safe path containment), runs the regression
   tests
   (`python -m unittest discover -s tests`), and fails if the views are stale. Edit
   `capabilities.json` only — never the generated files.

5. Open a PR. In the description, name the exact device, OS build, Xcode/SDK version and MLX
   version (they are in the receipt), and whether the machine ran on battery or AC.

## Rules

- **No cell without a receipt.** Corrections, additions and removals all need one — and it
  must be a supported receipt version (3, or 2 under the documented historical-inference
  policy) with completion evidence. An interrupted run (`complete: false`) is preserved in
  `receipts/` as evidence but can never back a cell. The map publishes observed rows only.
- **One chip per receipt.** Do not average machines or extrapolate across generations.
- **A failure is a result.** If a lane fails its check, fails to build, or is unavailable,
  that exact status belongs in the receipt; do not fabricate or `inferred`-tag a number.
- **Scope is capabilities**, not implementations: no model benchmarks and no vendor
  speculation; the probe kernels in `probes/` are the point. Unmeasured cells stay absent;
  the reserved `reported`/`inferred` tags are for a future source-linked representation, not
  a workaround.
- **Privacy**: the receipt is the artifact. Do not include process listings, hostnames,
  usernames, absolute home paths or raw logs in PRs, commit messages or attachments; the
  probes sanitize what they write, nothing else is sanitized for you.
- **Licensing**: by contributing you agree the split in LICENSES.md applies — scripts under
  MIT, data (receipts, the JSON map) and text under CC BY 4.0.
- Small PRs are welcome; there is no minimum number of rows.

## Review

Maintainers check that each receipt's device identity and check fields match the claim, and
that the protocol in METHODS.md was followed. Questions go in the PR; measured answers win.
