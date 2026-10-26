# Phase 0b: uncaptured, reproducible baseline (implementation scaffold)

This is a *standalone, source-controlled* experiment under the approved
[Phase 0 protocol](PHASE-0.md) and [research plan](PLAN.md).
Code review and CPU-only synthetic tests may proceed before an M5 Max is
available. **No baseline measurements have been collected by this PR.**

The script [tools/phase0b_baseline.py](../../tools/phase0b_baseline.py) is
intentionally separate from the published Precisit sweep protocols and their
receipt validator. Do not put its JSON under `receipts/`. It is published
and versioned to enable a reviewer to reproduce the baseline; its **SHA-256
is embedded in each local JSON result**, together with hashes of both upstream
probe helpers. Raw experimental JSON is gitignored by `experiments/`.

## On a real M5 Max

Use a quiet machine, one GPU experiment at a time, stable power source, and
the same Xcode / MLX version as the historical receipts where feasible
(**Xcode 27A266a, MLX 0.32.3**). The published M5 Max reference is a
**40-GPU-core, 128-GB** configuration; report a different SKU separately.
Never disable unrelated jobs or change system power settings without approval.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'mlx==0.32.3' 'numpy>=2,<3'
mkdir -p experiments/m5-max

# Full uncaptured FP16 rate-shape baseline:
python tools/phase0b_baseline.py \
  --shape rate --output experiments/m5-max/fp16-rate-001.json

# Optional same-lane, alternate-tile A/B control (both validated):
python tools/phase0b_baseline.py --shape rate \
  --tile-a 0 --tile-b 2 \
  --output experiments/m5-max/fp16-tiles-001.json

# Fast protocol smoke test only; NOT a representative NA benchmark:
python tools/phase0b_baseline.py --shape check \
  --output experiments/m5-max/fp16-check-001.json
```

`--shape rate` reuses Precisit's `ls.check_tile()` at CHK and RATE with a
single seeded RNG per tile. It performs NumPy correctness checks **before**
recording any time sample. No GPU capture occurs, and any
`MTL_CAPTURE_ENABLED`, shader validation, debug layer or device-wrapper
instrumentation is rejected before benchmarking.

## What is measured

- **Warm-up:** repeated synchronized batches, default `N=8`. Each
  time window records host-observed milliseconds per dispatch. After at
  least 0.5 seconds, the most recent three windows must differ by at
  most 2% of their median, or warm-up ends nonconverged at the configured
  8-second cap. These are *proposed initial criteria*, not validated
  frequency-stability rules.
- **Batching:** `N=1,2,4,8,16` (subject to output-memory budget). Each
  sample constructs N distinct output arrays before **one** `mx.eval(*outs)`
  synchronization. Default 512 MiB output-budget cap, further bounded by
  10% of reported RAM; rejected N values and reasons are recorded.
- **Fixed-cost decomposition:** host wall-clock `T(N)=a+b·N` regression
  with residual-based standard errors. Interpret `a` as *effective fixed
  cost*, `b` as *back-to-back host-observed time per dispatch* (NOT
  isolated GPU latency); kernel scheduling/overlap and queue overhead may
  affect both.
- **Order:** 7 repetitions by default, alternating ascending/descending
  batch sizes. Optional A/B tile test alternates A–B and B–A per
  repetition and provides distributions/medians and chronological data.
  A/B uses the **same lane and shape**, varying only tile geometry.
- **Nonconvergence:** results remain in local JSON and carry
  `steady_state_claim_allowed=false`; neither the rate nor tile choice
  should be published as a steady-state finding.

The script never runs `sudo`, changes power configuration, uploads a GPU
trace, writes an upstream receipt or modifies `matrix/`. GPU frequency
and Neural Accelerator counters are *not* measured. If desired,
`sudo powermetrics --samplers gpu_power` is a **separate manual step**
requiring machine-owner approval.

## After running

Review the output locally and post only a **sanitized summary** to
[issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1),
including the reviewed source URL, its SHA-256, chip/SKU and core count,
Xcode/Metal/MLX versions, power mode, warm-up convergence, selected N,
estimated `a,b` and their uncertainty, N-batch distributions and
unresolved limitations. Do not paste raw logs, usernames, home directories,
hostnames, profiler traces or memory contents.

Acceptance before the future reusable PR A: the finalized PR A runner should
reproduce the converged FP16 Phase 0b rate and `b` within the **predeclared
±5%** under matched hardware conditions, or within overlapping repeat-run
uncertainty; otherwise investigate the discrepancy. Phase 0b itself does
not satisfy the separate Phase 0a **M5 Max Xcode replay/counter gate**.
