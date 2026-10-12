# Probes

- `lane_sweep.py` — float and mixed lanes; produces `lane_sweep.json`.
- `integer_lanes.py` — integer and control lanes (plus its own `f16*f16` baseline); produces
  `integer_lanes.json` (pass `--json`).

See METHODS.md for the protocol and its traps, and README.md for how to contribute a receipt.
Both probes need the Xcode 27 SDK (see METHODS.md — MLX's injected headers gate the build).

Run on an Apple Silicon Mac:

```sh
pip install mlx numpy
python lane_sweep.py
python integer_lanes.py --json integer_lanes.json
```

`lane_sweep.py` writes `lane_sweep.json` next to the script; `integer_lanes.py` writes to the
path you pass to `--json` (relative to your working directory) and prints the summary without
one. Place the files under `receipts/<chip>/` when contributing.
