# Methods

Everything in `matrix/` comes from probes in `probes/`, run on real hardware. This file is
the protocol and the traps, so numbers from different contributors are comparable and the
claims stay scoped.

## The protocol

`lane_sweep.py` runs each lane through `mpp::tensor_ops::matmul2d` with operands in device
memory:

- shape 2048×4096×4096 (M×K×N), fixed;
- tile configurations: the standard sweep tries five; two warm-up calls; five timed calls per
  tile; the best result is kept (a burst figure). Records differ — each receipt lists the
  tiles its lanes actually ran (some seeded integer lanes ran one);
- `matmul2d_descriptor::mode::multiply` — never `multiply_accumulate`;
- correctness first: constant encodings (1.0 × 2.0) must produce the expected product;
  scaled lanes also run a per-n/per-kb block-scale check; a lane that fails its checks is
  reported `wrong` and its rate is not used;
- a receipt (`lane_sweep.json`) records device identity, MLX version, per-lane status, rates
  per tile, and the check results.

## Run conditions

Benchmark on a quiet, thermally settled machine; run one benchmark at a time; note AC/battery
and low-power mode in the PR description; let the machine cool between sweeps. Background GPU
work distorts a burst figure, and a fanless machine's sustained rate is not its burst rate.

## Traps (each one cost us rework)

- **`multiply_accumulate` with a recycled output buffer silently adds old results into new
  ones.** It produced exact-2×/3× "anomalies" in an earlier sweep. Use `mode::multiply`; if
  you deliberately accumulate, initialize the destination and say so.
- **Relaxed precision differs per lane.** The scaled lanes run with different
  relaxed-precision settings; a scaled-vs-plain comparison is a comparison of the tested
  paths, not of the scale bytes alone.
- **Warm-up matters.** First calls include pipeline compilation. The protocol warms up, and
  still keeps the best of several samples as a burst figure.
- **Burst is not sustained.** These are short probes. A fanless machine declines under
  sustained load; do not read a lane rate as a model rate.
- **Denominators.** All `vs_f16` values divide the same receipt's `f16*f16->f32` at the same
  shape and tile policy. Do not compare across shapes, tiles, or different scripts without
  saying so.
- **Whole-GPU vs per-core.** Absolute GOPS are whole-GPU rates for the measured SKU. Dividing
  by `gpu_cores` gives an approximate per-core figure (the CSV carries one), useful for
  comparing machines, but per-core scaling is not exact: SKUs differ in core count and memory,
  and those were not measured here. The `vs_f16` ratios are same-chip and cancel core count.
- **SKUs.** Record the exact configuration (GPU cores, memory) of the measured machine in the
  chip entry. A 8-core M5 Air and a 10-core M5 Air are different rows as far as absolutes go.
- **Availability is gated by the toolchain, not only the chip.** The tensor ops need recent
  Xcode/SDK headers, and MLX's injected kernel headers are a second gate: with Xcode 26.6,
  MLX 0.32.3 fails to compile every lane (`metal_fp8_e4m3_format` exists only in the Xcode 27
  Metal standard library), so pre-Xcode-27 hosts cannot produce any receipt with these
  probes. When lanes do compile, a lane may still fail to build or run at fp16 speed (no
  acceleration) rather than failing. Record which it is; all of these are results.
- `integer_lanes.py` is the companion sweep for integer and control lanes (it also re-measures
  `f16*f16` as its own baseline). Its receipt is `integer_lanes.json`; the same rules apply.
- **`MLX_ENABLE_TF32` does not touch these lanes** (they are fp16/bf16/8-bit operand probes,
  not float32 matmuls). State the setting you ran with.
- **`->f32` names the declared destination type, not established accumulation precision.** The
  float-input intrinsic family accumulates in 32-bit float, but the ground-truth checks here
  use constant encodings, and `relaxed_precision` allows operand mantissas to be truncated
  before the multiply (f32×f32 operands are admitted only on the relaxed path on the 16×16
  cooperative tensor). Do not read `->f32` as full-precision fp32 arithmetic.
- **Receipts name the probe that produced them.** Pre-publication receipts from the authoring
  project are named `p19b_mx_sweep` and `p23_mixed_fp8_lanes`; they are earlier versions of
  `lane_sweep.py` running the same protocol. An unfamiliar probe name means a legacy run —
  say so rather than reconstructing it.

## What a rate means

A row says: this GPU, this toolchain, this shape, this tile policy — measured at this rate.
It does not say how a model will run, what a kernel's overhead will cost, or what the units
physically are. Where a row is `inferred` or `reported`, the JSON says so and links the
source.
