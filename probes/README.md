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

The receipts land next to the scripts; place them under `receipts/<chip>/` when contributing.
