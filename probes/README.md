# Probes

- `lane_sweep.py` — the lane sweep that produces `lane_sweep.json`. See METHODS.md for the
  protocol and its traps, and README.md for how to contribute a receipt.

Run on an Apple Silicon Mac:

```sh
pip install mlx numpy
python lane_sweep.py
```

The receipt lands next to the script; place it under `receipts/<chip>/` when contributing.
