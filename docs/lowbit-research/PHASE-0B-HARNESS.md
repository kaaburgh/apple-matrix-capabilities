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

# 1. Ramp reproduction: upstream 5-sample protocol followed by convergence
#    and T(N). Run this FIRST after the correctness check.
python tools/phase0b_baseline.py --shape rate --upstream-replica \
  --output experiments/m5-max/fp16-rate-replica-001.json

# Allow machine to cool/settle before the next run.
# 2. Clean baseline without the upstream replica's extra GPU warm-up.
python tools/phase0b_baseline.py --shape rate \
  --output experiments/m5-max/fp16-rate-001.json

# Allow machine to cool/settle again.
# 3. Phase 2 same-lane, matched-precision tile contrast (a):
#    64x64x2 vs 64x32x4.
python tools/phase0b_baseline.py --shape rate \
  --tile-a 1 --tile-b 3 \
  --output experiments/m5-max/fp16-tiles-1v3-001.json

# Optional: compare 64x64x2 against 128x128x8 (both valid at RATE).
python tools/phase0b_baseline.py --shape rate \
  --tile-a 1 --tile-b 4 \
  --output experiments/m5-max/fp16-tiles-1v4-001.json

# Fast protocol smoke test only; NOT a representative NA benchmark:
python tools/phase0b_baseline.py --shape check \
  --output experiments/m5-max/fp16-check-001.json
```

`--shape rate` reuses Precisit's `ls.check_tile()` at CHK and RATE with a
single seeded RNG per tile; the CHK correctness check always uses upstream's canonical `TILES[0]`, while RATE checks the selected tile. Unsupported tile/shape divisibility fails *before any GPU work*. It performs NumPy correctness checks **before**
recording any time sample. No GPU capture occurs, and any
`MTL_CAPTURE_ENABLED`, shader validation, debug layer or device-wrapper
instrumentation is rejected before benchmarking.

## What is measured

- **Warm-up:** repeated synchronized batches, default `N=8`. **Every batch** records its offset from warm-up start, `N`, `construct_ms` and `eval_ms`, preserving the first few milliseconds of the ramp. Each window also records average `eval_ms` per dispatch. After at
  least 0.5 seconds, the most recent three windows must differ by at
  most 2% of their median, or warm-up ends nonconverged at the configured
  8-second cap. These are *proposed initial criteria*, not validated
  frequency-stability rules.
- **Batching:** `N=1,2,4,8,16` (subject to output-memory budget). Each
  sample constructs N distinct lazy output arrays **before the synchronization timer**, then times one `mx.eval(*outs)` invocation. Both the graph-construction and eval/synchronization durations are reported separately. This avoids systematically including Python graph-building time in the T(N) slope. Default 512 MiB output-budget cap, further bounded by
  10% of reported RAM; rejected N values and reasons are recorded.
- **Fixed-cost decomposition:** host-observed **evaluation/synchronization** time `T(N)=a+b·N` regression. The pooled OLS is secondary; the primary uncertainty descriptor is the set of **per-repetition slopes**, their median, min/max and descriptive 2.5–97.5% percentiles (not a calibrated confidence interval). Interpret `a` as *effective fixed
  cost*, `b` as *back-to-back host-observed time per dispatch* (NOT
  isolated GPU latency); kernel scheduling/overlap and queue overhead may
  affect both. MLX command-buffer limits (`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`) are recorded and may cause discontinuities.
- **Order:** 7 repetitions by default, alternating ascending/descending
  batch sizes. Each per-repetition slope records `order=asc|desc`;
  the results also include median `b` for each order and the desc-minus-asc
  difference, so within-run drift can be spotted. Optional A/B tile test
  alternates A–B and B–A per
  repetition and provides distributions/medians and time-stamped chronological data.
  A/B uses the **same lane and shape**, varying only tile geometry.
- **Upstream ramp replica (first recommended run):** `--upstream-replica` runs Precisit's own `time_tile(ctx, tile, reps=5, warmups=2)` *immediately after RATE correctness* to capture first-sample behavior. It changes the subsequent warm-up initial state, so compare replica-on and replica-off runs separately. Give the GPU time to return to comparable conditions between runs.
- **Nonconvergence:** results remain in local JSON and carry
  `steady_state_claim_allowed=false`; neither the rate nor tile choice
  should be published as a steady-state finding.

On success, local JSON includes provenance and complete raw sample series; on blocked/failed attempts a sanitized **status/stage/error** JSON is written whenever the valid output directory is accessible. Exit codes: 0 observed, 1 failed, 2 blocked. Raw error details stay on local stderr. Both outcome types serialize *before* atomic no-overwrite publication, so invalid floating-point JSON cannot leave a partial result file.

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
estimated `a,b`, per-repetition slope spread, separate construction cost, warm-up first-batch samples, and
unresolved limitations. Do not paste raw logs, usernames, home directories,
hostnames, profiler traces or memory contents.

Acceptance before the future reusable PR A: the finalized PR A runner should
reproduce the converged FP16 Phase 0b rate and `b` within the **predeclared
±5%** under matched hardware conditions, or within comparable observed repeat-run
spread; otherwise investigate the discrepancy. Phase 0b itself does
not satisfy the separate Phase 0a **M5 Max Xcode replay/counter gate**.
