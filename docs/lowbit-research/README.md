# Low-bit TensorOps architecture research

**Status:** approved research plan (2026-10-25). Implementation is tracked separately in pull requests.

This fork studies GPU execution of FP16/BF16, FP8, FP4 and integer TensorOps on Apple Silicon. The project aims to distinguish observed performance, Neural Accelerator activity, format conversion overhead and claims about undocumented hardware. **GPU instruction encodings / ISA reconstruction are not in v1 scope.**

- [Full approved plan](PLAN.md): objectives, constraints, review findings, acceptance criteria, PR ordering and evidence requirements.
- [Phase 0b baseline harness](PHASE-0B-HARNESS.md): reproducible uncaptured convergence and N-batch methodology (source/test scaffold; no GPU data yet).
- [Phase 0 protocol](PHASE-0.md): executable checklist for the hardware/toolchain feasibility spike and an independently reproducible uncaptured baseline.
- [GitHub issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1): original discussion and review history. The repository documents are the versioned implementation reference.
- [Upstream measurement methodology](../../METHODS.md): do not alter the stock protocol, published receipts or capability-matrix interpretation without an explicit separate change.

## Delivery checkpoints

| Sequence | Work | Hard gate |
|---|---|---|
| 0a | Real-device capture/replay and profiler counter inventory | Trace identifiable; at least one useful observed profiling signal |
| 0b | Disposable uncaptured convergence and N-batch baseline | Sanitized source, hash, method and numerical summary retained |
| PR A | Isolated float-lane capture and batched Instruments loop | CPU safety tests + real Apple GPU smoke test; reproduce Phase 0b FP16 |
| Manual gate | Calibrated NA/counter findings from float controls | Counters discriminating or explicitly labelled otherwise |
| PR B | INT8 adapter, controlled comparisons | Matched tile & timing and INT8 calibration |
| PR C | Named FP4/scale/layout ablations | Frozen historical seeds, validated schema-3 experimental receipts |
| PR D (optional) | Profiling sidecars/export | Only after actual counter availability/export is verified |

**Hardware boundary:** a CI pass on Linux or a CPU stub does *not* demonstrate Metal compilation, successful `.gputrace` replay, NA utilization or performance. Raw traces and process/system diagnostics must remain local.
