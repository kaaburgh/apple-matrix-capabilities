# Probes

- `lane_sweep.py` — float and mixed lanes; writes `lane_sweep.json` in the working directory
  by default (`--json PATH`).
- `integer_lanes.py` — integer and control lanes (plus its own `f16*f16` baseline); writes
  `integer_lanes.json` by default (`--json PATH`).

Both validate every timed tile against a decoded-operand NumPy reference before timing
(receipt schema 2), and refuse to overwrite an existing receipt unless `--force` is passed —
receipts are evidence, prefer a new name. See METHODS.md for the protocol and its traps, and
README.md for how to contribute a receipt. Both probes need the Xcode 27 SDK (see METHODS.md —
MLX's injected headers gate the build).

Run on an Apple Silicon Mac:

```sh
pip install mlx numpy
python lane_sweep.py
python integer_lanes.py --json integer_lanes.json
```

Place the receipts under `receipts/<chip>/` when contributing.
