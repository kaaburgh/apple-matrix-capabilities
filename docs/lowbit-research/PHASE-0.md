# Phase 0: hardware feasibility and uncaptured baseline

**Plan:** [PLAN.md](PLAN.md). **Review and evidence:** [issue #1](https://github.com/kaaburgh/apple-matrix-capabilities/issues/1).

This is a runbook for the accepted **Phase 0a** (disposable feasibility spike) and **Phase 0b** (uncaptured, disposable experiment). It is not proof that either has run. Results requiring a physical Apple Silicon Mac remain blocked until collected on hardware.

## Phase 0a — capture and counter feasibility

- [ ] Record exact chip/SKU (published M5 Max: 40 GPU cores, 128 GB), GPU core count, RAM, macOS build, Xcode and Metal compiler, MLX exact version/build, power source, and repository/source hashes. Use **MLX 0.32.3** to compare against the published receipts; deviations must be flagged.
- [ ] On a real Mac, run the stock FP16 TensorOps path and its correctness check. Ensure full **Xcode 27** and compatible SDK; do not claim an Xcode command-line-tools-only installation is sufficient.
- [ ] Set `MTL_CAPTURE_ENABLED=1` **before starting Python**; use the *documented installed-MLX* Metal capture API around one previously validated dispatch. The `.gputrace` output path must not already exist.
- [ ] Open and replay the trace in **Xcode Metal Debugger**. Record expected/observed dispatch name and count and Xcode **replay** settings (capture-time GPU state does not determine replay-time performance). Determine if stock MLX gives source attribution; if not, try a debug build of the **same upstream MLX tag**, not an experimental MLX fork.
- [ ] Separately use **Instruments Metal System Trace** to determine which timeline counters are visible; record exact names, scope and granularity of Neural Accelerator utilization, instruction mix, performance limiters, memory and GPU state. A `.gputrace` and Instruments `.trace` are different evidence sources.
- [ ] Positive control: FP16 `mpp::tensor_ops::matmul2d`. Negative control: separately implemented FP16 `simdgroup_matrix` (if supported). Keep the latter's tested source, hash and numerical correctness result.
- [ ] Check whether **documented** `xctrace` export can extract the needed counters. Do not design automated exports until verified.

**Stop gate:** no reusable capture infrastructure before a real capture has been replayed, its dispatch identified and at least one useful signal inspected. If unavailable, post the observed blocker and revise scope. Never equate NA utilization with physical FP4/FP8 ISA support.

## Phase 0b — baseline, measurement bias and convergence

- [ ] Preserve original `receipts/`. A stock-probe repeat may generate a **new** receipt, never overwrite historical data; disposable convergence/T(N) measurements are **not receipt-schema data**.
- [ ] Use a small independent script, stored initially under **gitignored `experiments/<chip>/`**. Before selecting PR A's settings, publish the script's **sanitized source** (via a reviewed repository file or gist), SHA-256 and summary in issue #1, allowing independent reruns.
- [ ] Validate the test output *before* timing; keep CPU NumPy work and all correctness checks outside measurement windows.
- [ ] Measure time-based back-to-back warm-up (`>=0.5 s` as an initial example) with bounded last-window convergence (~2% as a proposal), and record convergence and the maximum allowed warm-up duration.
- [ ] Run `N=1,2,4,8,16` independent outputs per synchronized batch; fit `T(N) = a + bN`, with fit uncertainty and output-allocation/cache-budget safeguards. Label `b` **back-to-back throughput**, not a single GPU dispatch duration.
- [ ] Alternate comparable A/B measurements after warm-up; record distributions, spread, rate and selection bias, rather than relying on best-of-five. GPU power/frequency collection is optional: `sudo powermetrics --samplers gpu_power` **only with the machine owner's approval**; no default process-list sampling.
- [ ] Treat `MTL_CAPTURE_ENABLED`, shader validation and debug layers as disqualifying for performance timing. Contrast stock-wheel and same-tag debug-build results *uncaptured* if a debug build is used.
- [ ] Establish an FP16 baseline with declared device/shape/tile, chosen N, warm-up result and fitted `a,b`, plus uncertainty. If nonconverged, retain as diagnostic data but do **not** label it steady state.
- [ ] Include source hashes, environment, GPU SKU, timestamp, and toolchain in the issue summary. Existing M5 Max receipts exhibit a suspected first-sample-to-last-sample ramp: **the mechanism is a hypothesis**, not established hardware behavior.

**Handoff to PR A:** re-implement reusable warm-up/batching in the production tool; reproduce Phase 0b FP16 rate and `b` within a **predeclared ±5%** under matched conditions or overlapping repeat-run uncertainty, otherwise investigate. For Metal Debugger capture, only short JIT/first-touch warm-up is needed; a replay is **never labelled converged**.

## Evidence template (post sanitized results to issue #1)

```text
Stage: 0a or 0b
Status: BLOCKED / FAILED / OBSERVED
Device, GPU-core count, RAM:
macOS build / Xcode / Metal compiler:
MLX exact version, wheel or debug build:
Probe and experiment script sources (SHA-256 / reviewed source URL):
Power source / low-power mode:
Lane / M,K,N / tile / precision / packing:
Capture: Xcode replay settings, dispatch identity, counter names & scopes
Live loop: batch N, convergence, idle fraction, profiler time window
T(N): a, b, confidence/uncertainty and memory limits
Counter calibration: positive / negative / discriminating / unavailable
Privacy review: raw traces kept locally; identifiers removed
Open questions / blockers:
```

Do not attach raw `.gputrace`, Instruments `.trace`, unrestricted process lists, home paths or credentials to this public repository.

**PR A handoff clarification:** the checked-in Phase 0a and Phase 0b utilities are maintained source dependencies of the reusable PR A runner, not throwaway files. Each manifest preserves hashes. A Phase 0b fixed-N median is comparable to a PR A fixed-N median on the same physical SKU/shape/tile; a fitted `b` requires a separate multi-N Phase 0b run plus a truly different hardware observation such as the Instruments GPU-busy interval. See [PR A runbook](PR-A-FLOAT-CAPTURE.md).
