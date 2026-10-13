#!/usr/bin/env python3
"""Integer and control lanes: measured matrix-unit rates on an Apple Silicon GPU.

Companion to `lane_sweep.py` (float and mixed lanes). Runs Metal Performance
Primitives `matmul2d` directly on device-memory tensors (no MLX tiling, no
staging) through stock MLX's `mx.fast.metal_kernel`, for several operand type
combinations and threadgroup tile shapes, at prefill-like GEMM shapes.

Rate is 2*M*K*N / wall time per call (multiply + add counted separately), so
integer lanes are reported in the same unit as the float sweep. See METHODS.md.

Correctness (receipt schema 2): checks use nonuniform random operands and a
NumPy reference built from the decoded operands; every tile is checked at the
small check shape and every timed (tile, shape) is checked at the timing shape
before it is timed. int4b packing is probed lo-first, with the other nibble
order tried only if the first fails; the matching order and the attempts are
recorded. Failures are recorded per lane/tile/shape with structured,
sanitized error objects; the receipt is written atomically per lane and
marked `complete` only at the end.

Requirements: an Apple Silicon Mac, MLX + NumPy, and an Xcode/SDK whose
MetalPerformancePrimitives headers provide the tensor ops (Xcode 27 SDK; see
METHODS.md for the toolchain gate).

Usage: python integer_lanes.py [--json integer_lanes.json] [--reps 7]
       [--calls 4] [--relaxed] [--shapes 2048x4096x4096,4096x4096x4096]
"""

import argparse
import hashlib
import json
import hashlib
import math
import os
import platform
import statistics
import sys
import time

import numpy as np

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

RECEIPT_SCHEMA = 3
SEED = 20261012


def cap_memory():
    try:
        mem = int(mx.device_info().get("memory_size", 0))
    except Exception:
        mem = 0
    limit = min(12 * 2**30, max(1 * 2**30, int(mem * 0.35))) if mem else 4 * 2**30
    mx.set_memory_limit(limit)
    mx.set_cache_limit(min(2 * 2**30, limit // 4))


cap_memory()

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
using namespace mpp::tensor_ops;
"""

# A is (M,K) row-major, B is (K,N) row-major, C is (M,N) row-major. Tensor
# extents list the innermost dimension first.
SRC = """
    constexpr auto desc = matmul2d_descriptor(
        {mt}, {nt}, static_cast<int>(metal::dynamic_extent), false, false, {relaxed});
    matmul2d<desc, metal::execution_simdgroups<{sg}>> op;
    auto tgx = (int)threadgroup_position_in_grid.x;
    auto tgy = (int)threadgroup_position_in_grid.y;
    auto tA = metal::tensor<device {act}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {acast}*)A, metal::dextents<int32_t, 2>({K}, {M}));
    auto tB = metal::tensor<device {wt}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {wcast}*)B, metal::dextents<int32_t, 2>({N}, {K}));
    auto tC = metal::tensor<device {acc}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {acc}*)C, metal::dextents<int32_t, 2>({N}, {M}));
    auto mA = tA.slice(0, tgy * {mt});
    auto mB = tB.slice(tgx * {nt}, 0);
    auto mC = tC.slice(tgx * {nt}, tgy * {mt});
    op.run(mA, mB, mC);
"""

# name: activation type, activation cast, weight type, weight cast, accumulator, out dtype
LANES = {
    "f16*f16->f32": ("half", "half", "half", "half", "float", mx.float32),
    "f16*f16->f16": ("half", "half", "half", "half", "half", mx.float16),
    "bf16*bf16->f32": ("bfloat", "bfloat", "bfloat", "bfloat", "float", mx.float32),
    "i8*i8->i32": ("int8_t", "int8_t", "int8_t", "int8_t", "int32_t", mx.int32),
    "f16*i8->f32": ("half", "half", "int8_t", "int8_t", "float", mx.float32),
    "f16*i4b->f32": ("half", "half", "metal::int4b_format", "uchar", "float", mx.float32),
    "i8*i4b->i32": ("int8_t", "int8_t", "metal::int4b_format", "uchar", "int32_t", mx.int32),
}

# (rows per threadgroup tile, columns per tile, simdgroups per threadgroup)
TILES = [(64, 32, 4), (64, 64, 2), (64, 64, 4), (64, 128, 4), (128, 128, 8)]
CHK = (128, 256, 256)

# Stable per-lane seed index: independent of the requested lane subset/order.
LANES_INDEX = {lane: i for i, lane in enumerate(LANES)}


def pack_nibbles(w, order):
    u = (w.astype(np.int16) & 0xF).astype(np.uint8)
    lo, hi = u[:, 0::2], u[:, 1::2]
    return (lo | (hi << 4)).astype(np.uint8) if order == "lo" else (hi | (lo << 4)).astype(np.uint8)


def has_i4b(lane):
    return "int4b" in LANES[lane][2]


def tol_for(lane):
    acc = LANES[lane][4]
    if acc == "int32_t":
        return "exact"
    if "bf16" in lane or lane.endswith("f16"):
        return 2e-2
    return 2e-3


def operands(lane, m, k, n, rng, small_ints, order):
    act, _, wt, _, _, _ = LANES[lane]
    lim = 8 if (small_ints or "int4b" in wt) else 127
    if act == "half":
        a_mx = mx.array(rng.standard_normal((m, k)).astype(np.float16))
        a_dec = np.array(a_mx.astype(mx.float32), dtype=np.float64)
    elif act == "bfloat":
        a_mx = mx.array(rng.standard_normal((m, k)).astype(np.float32)).astype(mx.bfloat16)
        a_dec = np.array(a_mx.astype(mx.float32), dtype=np.float64)
    else:
        a_mx = mx.array(rng.integers(-lim, lim, size=(m, k)).astype(np.int8))
        a_dec = np.array(a_mx).astype(np.float64)
    if "int4b" in wt:
        w = rng.integers(-8, 8, size=(k, n)).astype(np.int8)
        b_mx = mx.array(pack_nibbles(w, order))
        b_dec = w.astype(np.float64)
    elif wt == "half":
        b_mx = mx.array(rng.standard_normal((k, n)).astype(np.float16))
        b_dec = np.array(b_mx.astype(mx.float32), dtype=np.float64)
    elif wt == "bfloat":
        b_mx = mx.array(rng.standard_normal((k, n)).astype(np.float32)).astype(mx.bfloat16)
        b_dec = np.array(b_mx.astype(mx.float32), dtype=np.float64)
    else:
        b_mx = mx.array(rng.integers(-lim, lim, size=(k, n)).astype(np.int8))
        b_dec = np.array(b_mx).astype(np.float64)
    mx.eval(a_mx, b_mx)
    return a_dec, b_dec, a_mx, b_mx


def build(lane, tile, m, k, n, relaxed):
    act, acast, wt, wcast, acc, _ = LANES[lane]
    mt, nt, sg = tile
    src = SRC.format(mt=mt, nt=nt, sg=sg, act=act, acast=acast, wt=wt, wcast=wcast,
                     acc=acc, M=m, K=k, N=n, relaxed="true" if relaxed else "false")
    tag = "".join(c if c.isalnum() else "_" for c in lane)
    name = f"amc_i_{tag}_m{mt}n{nt}s{sg}_{m}x{k}x{n}_{'rx' if relaxed else 'ex'}"
    return mx.fast.metal_kernel(name=name, input_names=["A", "B"], output_names=["C"],
                                header=HEADER, source=src)


def call(kern, lane, tile, a_mx, b_mx, m, n):
    mt, nt, sg = tile
    return kern(inputs=[a_mx, b_mx], grid=(sg * 32 * (n // nt), m // mt, 1),
                threadgroup=(sg * 32, 1, 1), output_shapes=[(m, n)],
                output_dtypes=[LANES[lane][5]])[0]


def compare(lane, out, a_dec, b_dec):
    ref = a_dec @ b_dec
    tol = tol_for(lane)
    if tol == "exact":
        return bool(np.array_equal(out.astype(np.int64), ref.astype(np.int64))), 0.0
    err = float(np.abs(out.astype(np.float64) - ref).max() / max(np.abs(ref).max(), 1e-30))
    return bool(err < tol), err


def check_tile(lane, tile, relaxed, rng, order):
    m, k, n = CHK
    a_dec, b_dec, a_mx, b_mx = operands(lane, m, k, n, rng, True, order)
    out = np.array(call(build(lane, tile, m, k, n, relaxed), lane, tile, a_mx, b_mx, m, n))
    ok, err = compare(lane, out, a_dec, b_dec)
    return {"stage": "check_shape", "rel_err": err, "tolerance": tol_for(lane), "ok": ok}


def rate_check(lane, tile, relaxed, a_dec, b_dec, a_mx, b_mx, shape):
    m, k, n = shape
    out = np.array(call(build(lane, tile, m, k, n, relaxed), lane, tile, a_mx, b_mx, m, n))
    ok, err = compare(lane, out, a_dec, b_dec)
    return ok, {"stage": f"rate_shape_{m}x{k}x{n}", "rel_err": err,
                "tolerance": tol_for(lane), "ok": ok}


def time_tile(lane, tile, relaxed, a_mx, b_mx, shape, reps, calls):
    m, k, n = shape
    kern = build(lane, tile, m, k, n, relaxed)

    def batch():
        return [call(kern, lane, tile, a_mx, b_mx, m, n) for _ in range(calls)]

    for _ in range(2):
        mx.eval(*batch())
    ts = []
    for _ in range(reps):
        outs = batch()
        t0 = time.perf_counter()
        mx.eval(*outs)
        ts.append((time.perf_counter() - t0) / calls)
        del outs
    return min(ts), statistics.median(ts), [round(t * 1e3, 3) for t in ts]


err_obj = C.err_obj
write_receipt = C.write_receipt


def machine_meta():
    dev = mx.device_info()
    try:
        with open(__file__, "rb") as f:
            probe_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        probe_hash = "unknown"
    device = {k: dev.get(k) for k in ("device_name", "architecture", "memory_size",
                                      "max_recommended_working_set_size")}
    device["gpu_cores"] = C.gpu_cores()
    return {
        "probe": "apple-matrix-capabilities/integer_lanes",
        "receipt_schema": RECEIPT_SCHEMA,
        "probe_sha256": probe_hash,
        "sources": C.source_hashes(C.probe_files(__file__)),
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": {"version": platform.mac_ver()[0],
               "build": C.first_line(C.run_text(["sw_vers", "-buildVersion"]))},
        "toolchain": C.toolchain_meta(),
        "power": C.power_meta(),
        "env": {"MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32", "unset"),
                "MLX_METAL_JIT": os.environ.get("MLX_METAL_JIT", "unset")},
        "device": device,
    }


def discover_packing(lane, tile, relaxed, rng):
    """Nibble order for int4b lanes: try `lo`, then `hi`. Returns (order, attempts).

    Raises `_common.PackingFailed` (carrying the orders actually attempted)
    when neither order passes or a probe raises; callers record that as
    terminal evidence instead of letting the process die.
    """
    attempts = []
    try:
        attempts.append("lo")
        if check_tile(lane, tile, relaxed, rng, "lo")["ok"]:
            return "lo", attempts
        attempts.append("hi")
        if check_tile(lane, tile, relaxed, rng, "hi")["ok"]:
            return "hi", attempts
        raise C.PackingFailed(attempts)
    except C.PackingFailed:
        raise
    except Exception as exc:
        raise C.PackingFailed(attempts, exc) from exc


def terminal_row(lane, tile, shape, order, attempts, status, stage,
                 checks=None, error=None):
    """One terminal outcome for one (lane, tile, shape) unit of requested work.

    Every unit gets exactly one row whether it timed, failed a check or could
    not be built, so receipt coverage can distinguish failure from
    interruption.
    """
    row = {"lane": lane, "tile": list(tile), "M": shape[0], "K": shape[1],
           "N": shape[2], "status": status, "stage": stage, "packing": order,
           "packing_attempts": list(attempts), "checks": checks or []}
    if error is not None:
        row["error"] = error
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="integer_lanes.json",
                    help="receipt path (default integer_lanes.json in the working directory)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing receipt")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--relaxed", action="store_true")
    ap.add_argument("--shapes", default="2048x4096x4096,4096x4096x4096")
    ap.add_argument("--lanes", default=None, help="comma-separated lane names, in run order")
    ap.add_argument("--tiles", default=None, help="e.g. 64x128x4,128x128x8")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.reps < 1 or args.calls < 1:
        print("--reps and --calls must be positive", file=sys.stderr)
        return 2
    try:
        shapes = [C.parse_int_tuple(s, "--shapes") for s in args.shapes.split(",")]
        tiles = ([C.validate_tile(C.parse_int_tuple(t, "--tiles")) for t in
                  args.tiles.split(",")] if args.tiles
                 else [C.validate_tile(t) for t in TILES])
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2
    lanes = args.lanes.split(",") if args.lanes else list(LANES)
    for lane in lanes:
        if lane not in LANES:
            print(f"unknown lane: {lane}", file=sys.stderr)
            return 2
    for m, k, n in shapes + [CHK]:
        for mt, nt, _ in tiles:
            if m % mt or n % nt:
                print(f"shape {m}x{k}x{n} is not divisible by tile {mt}x{nt}", file=sys.stderr)
                return 2
    if os.path.exists(args.json) and not args.force:
        print(f"{args.json} exists; pass --force to overwrite "
              "(receipts are evidence - prefer a new name)", file=sys.stderr)
        return 2

    out = machine_meta()
    out.update({"relaxed": args.relaxed, "calls": args.calls, "reps": args.reps,
                "protocol": {"shapes": [f"{m}x{k}x{n}" for m, k, n in shapes],
                             "check_shape": f"{CHK[0]}x{CHK[1]}x{CHK[2]}",
                             "tiles": [f"{t[0]}x{t[1]}x{t[2]}" for t in tiles],
                             "lanes_requested": list(lanes),
                             "warmups": 2, "seed": args.seed, "reps": args.reps,
                             "sample": "reps samples of one batched timer over `calls` "
                                       "kernel calls; call construction before the timer",
                             "packing_checked": "lo-first; the other nibble order is "
                                                "tried only if the first fails"}})
    rows = []
    out["rows"] = rows
    out["complete"] = False
    write_receipt(out, args.json)
    for lane in lanes:
        rng = np.random.default_rng([args.seed, LANES_INDEX[lane]])
        packed = has_i4b(lane)
        order, attempts = None, []
        if packed:
            try:
                order, attempts = discover_packing(lane, tiles[0],
                                                   args.relaxed, rng)
            except Exception as exc:
                attempts = list(getattr(exc, "attempts", attempts))
                print(f"[local] {lane}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                err = err_obj(exc, "packing_check")
                for tile in tiles:
                    for shape in shapes:
                        rows.append(terminal_row(lane, tile, shape, None, attempts,
                                                 "failed", "packing_check",
                                                 error=err))
                print(f"{lane:16} PACKING CHECK FAILED {type(exc).__name__}",
                      flush=True)
                write_receipt(out, args.json)
                continue
        for tile in tiles:
            try:
                chk = C.finite_check(check_tile(lane, tile, args.relaxed, rng, order))
            except Exception as exc:
                print(f"[local] {lane} {tile}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                err = err_obj(exc, "check_shape")
                for shape in shapes:
                    rows.append(terminal_row(lane, tile, shape, order, attempts,
                                             "failed", "check_shape", error=err))
                print(f"{lane:16} tile {tile}: CHECK FAILED {type(exc).__name__}",
                      flush=True)
                write_receipt(out, args.json)
                continue
            if not chk["ok"]:
                for shape in shapes:
                    rows.append(terminal_row(lane, tile, shape, order, attempts,
                                             "wrong", "check_shape", checks=[chk]))
                print(f"{lane:16} tile {tile}: WRONG at check shape", flush=True)
                write_receipt(out, args.json)
                continue
            for shape in shapes:
                m, k, n = shape
                try:
                    a_dec, b_dec, a_mx, b_mx = operands(lane, m, k, n, rng, False, order)
                    ok, rchk = rate_check(lane, tile, args.relaxed, a_dec, b_dec,
                                          a_mx, b_mx, shape)
                    rchk = C.finite_check(rchk)
                    if not ok:
                        rows.append(terminal_row(
                            lane, tile, shape, order, attempts, "wrong", "rate_shape",
                            checks=[chk, rchk],
                            error={"phase": "rate_check", "type": "WrongResult",
                                   "category": "wrong_result"}))
                        print(f"{lane:16} tile {tile}: WRONG at {m}x{k}x{n}",
                              flush=True)
                        write_receipt(out, args.json)
                        continue
                    best, med, samples = time_tile(lane, tile, args.relaxed, a_mx, b_mx,
                                                   shape, args.reps, args.calls)
                    if (best <= 0 or not math.isfinite(best) or not math.isfinite(med)
                            or any(not math.isfinite(s) for s in samples)):
                        raise ValueError(f"non-finite or non-positive timing "
                                         f"(best={best!r}, med={med!r})")
                    g = lambda t: 2.0 * m * k * n / t / 1e9
                    row = terminal_row(lane, tile, shape, order, attempts, "ok",
                                       "timed", checks=[chk, rchk])
                    row.update({"ms_best": best * 1e3, "ms_med": med * 1e3,
                                "ms_samples": samples, "gops_best": g(best),
                                "gops_med": g(med)})
                    rows.append(row)
                    print(f"{lane:16} tile m{tile[0]}n{tile[1]}s{tile[2]}  {m}x{k}x{n}  "
                          f"best {g(best):8.0f}  median {g(med):8.0f} G(FL)OP/s  "
                          f"err {rchk['rel_err']:.1e}  packing={order}", flush=True)
                    del a_mx, b_mx
                except Exception as exc:
                    print(f"[local] {lane} {tile} {m}x{k}x{n}: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    rows.append(terminal_row(lane, tile, shape, order, attempts,
                                             "failed", "timing", checks=[chk],
                                             error=err_obj(exc, "timing")))
                    print(f"{lane:16} tile {tile} shape {m}x{k}x{n}: "
                          f"TIMING FAILED {type(exc).__name__}", flush=True)
                    write_receipt(out, args.json)
        write_receipt(out, args.json)
    out["results"] = {
        "rows_ok": sum(1 for r in rows if r["status"] == "ok"),
        "rows_wrong": sum(1 for r in rows if r["status"] == "wrong"),
        "rows_failed": sum(1 for r in rows if r["status"] == "failed"),
        "rows_total": len(rows)}
    out["complete"] = True
    write_receipt(out, args.json)
    print(f"receipt: {args.json} ({out['results']})")
    return 0 if out["results"]["rows_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
