# Methods

Everything in `matrix/` comes from probes in `probes/`, run on real hardware. This file is
the protocol and the traps, so numbers from different contributors are comparable and the
claims stay scoped.

## The protocol

`lane_sweep.py` (float and mixed lanes) and `integer_lanes.py` (integer and control lanes,
plus its own f16 baseline) run each lane through `mpp::tensor_ops::matmul2d` with operands in
device memory:

- shapes: 2048×4096×4096 (M×K×N) for the standard sweep (the integer probe also offers
  4096×4096×4096); a small 64×1024×256 (float) / 128×256×256 (integer) shape is used for a
  fast wiring check;
- tile configurations: the standard sweep tries five (some seeded schema-1 rows ran one);
  each receipt lists the tiles its lanes actually ran;
- correctness first: deterministic nonuniform, sign-varying **values** that are exactly
  representable in every lane format (and under any mantissa truncation) are drawn, encoded
  through the same table the probe uses, and the NumPy reference multiplies those intended
  values. The check compares the kernel's output against that reference; it does not
  independently decode the packed operand buffer. Every timed tile is validated at the timing
  shape before it is timed, packed formats are probed lo-first with the other nibble order
  tried only if the first fails (the matching order and the orders actually attempted are
  recorded), and a lane that fails a check is reported `wrong` and never timed;
- `matmul2d_descriptor::mode::multiply` — never `multiply_accumulate`;
- timing procedure differs by probe and each receipt records it: the float sweep takes five
  single-call samples with the kernel-call construction inside the timer; the integer sweep
  takes seven samples of one batched timer over four kernel calls constructed before it;
  warm-ups precede both; best-of-samples and median are kept per tile;
- receipts (versions 2 and 3) record probe name and SHA-256, the hashes of every source file
  the probe depends on (`sources`, including `probes/_common.py`), Python/NumPy/MLX versions,
  OS build, Xcode/Metal toolchain including the Xcode build line, power source and low-power
  mode, the device fields the map uses, protocol (shapes, tiles, warm-ups, reps, seed) and per
  lane: status, `relaxed_precision`, packing order and attempts, structured check results and
  per-tile samples. Receipts are written atomically and end with `complete: true`.
  Schema 3 additionally records the **requested work** (`lanes_requested`; integer rows carry
  M/K/N and a stage for every terminal outcome, success or failure), so completion is compared
  against the request. Schema-2 receipts predate that field: their completion is only
  *inferred* from the lanes/tiles/shapes that survived, which cannot prove a requested item
  that left no terminal row; the generated receipt index labels that `coverage (inferred,
  v2)`. A receipt with `complete: false` is an interrupted checkpoint that cannot back a cell.
  Failure objects in schema-3 receipts carry a controlled category plus the exception type —
  raw exception text stays in the contributor's local console log, never in the receipt. Schema 1 receipts (no `receipt_schema` field) are preserved as-is; their
  checks used constant encodings, and they are background evidence that no cell may cite.
  `receipts/schema.json` defines the accepted formats, and CI validates **every** receipt under
  `receipts/`, cited or not.

`geometry_sweep.py` is the opt-in companion for the geometry dimension: the same lanes,
checks and timing procedure, run across a fixed list of real prefill shapes (wide-N, deep-K,
low-rank, narrow families; M 512/2048/8192; the list lives in the script) instead of the
single standard shape. It writes `geometry_sweep.json` with per-shape, per-tile rates and
checks, per-lane precision and nibble-order metadata, and the same protocol block. The standard-shape columns stay the default everywhere; geometry receipts are
additional evidence, one per chip. The SHA-256 identifies the probe revision that produced the receipt and
  stops matching once the probe changes; that is expected, and receipts keep citing their own
  revision.

## Run conditions

Benchmark on a quiet, thermally settled machine; run one benchmark at a time; note AC/battery
and low-power mode in the PR description; let the machine cool between sweeps. Background GPU
work distorts a burst figure, and a fanless machine's sustained rate is not its burst rate.
On a machine shared with other users or agents, sample `ps` and the load average before the
run and periodically while it runs, and re-run rather than publish if anything else was
active — a receipt cannot prove the machine was alone. Those process/load logs are **local
run notes only**: keep them out of the repository and PRs. Receipt errors are sanitized by the
probe; anything you attach yourself (screenshots, `ps` output, logs) is not.

## Traps (each one cost us rework)

- **`multiply_accumulate` with a recycled output buffer silently adds old results into new
  ones.** It produced exact-2×/3× "anomalies" in an earlier sweep. Use `mode::multiply`; if
  you deliberately accumulate, initialize the destination and say so.
- **Relaxed precision differs per lane.** The scaled lanes run with different
  relaxed-precision settings; a scaled-vs-plain comparison is a comparison of the tested
  paths, not of the scale bytes alone. Receipt schema 2 records the flag per lane:
  unscaled lanes run relaxed on unless the lane name carries the `strict_` marker, scaled
  lanes run relaxed off unless it carries the `relaxed_` marker. Use the `strict_`/`relaxed_`
  variants to separate the scale plane from the relaxed flag.
- **Packed nibble order is inferred, not assumed.** For packed formats the probe tries the
  `lo` order first and only tries `hi` if `lo` fails a check; it records which order matched
  (`packing`) and which orders were attempted (`packing_attempts`). Do not carry a convention
  across toolchains or formats, and do not read a `packing` value as "both orders were
  validated" when only `lo` ran.
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
- **`MLX_ENABLE_TF32` and the f32 lanes.** Most lanes are fp16/bf16/8-bit operand probes that
  the flag does not touch, but the `f32*f32->f32` and `strict_f32*f32->f32` lanes do run
  float32 operand matmuls through the tensor op; their precision is governed by the tensor
  op's `relaxed_precision` (recorded per lane), and the receipts record the environment's
  `MLX_ENABLE_TF32` value (`unset` in the seed runs). State the setting you ran with.
- **`->f32` names the declared destination type, not established accumulation precision.**
  Do not read `->f32` as full-precision fp32 arithmetic. Current checks compare kernel output
  against the intended-value reference under the lane's recorded `relaxed_precision`; seed
  receipts measured before the schema-2 checks used constant encodings, which constrains the
  error signal but not the accumulation behavior. No internal cooperative-tensor path or
  accumulation width is asserted here beyond what the per-lane flag and checks record.
- **Receipts name the probe that produced them.** Pre-publication receipts from the authoring
  project are named `p19b_mx_sweep` and `p23_mixed_fp8_lanes`; they are earlier versions of
  `lane_sweep.py` running the same protocol. An unfamiliar probe name means a legacy run —
  say so rather than reconstructing it.

## What a rate means

A row says: this GPU, this toolchain, this shape, this tile policy — measured at this rate.
It does not say how a model will run, what a kernel's overhead will cost, or what the units
physically are.

The matrix currently represents **observed** rows only: every published cell cites a schema-2
receipt, and the validator rejects a chip carrying measured lanes under any other evidence tag.
`reported` and `inferred` are reserved for future rows with an external source link and are
unused today; chips without receipts are listed in `capabilities.json`'s `unknown_chips`.
