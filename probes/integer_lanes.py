#!/usr/bin/env python3
"""Integer and control lanes: measured matrix-unit rates on an Apple Silicon GPU.

Companion to `lane_sweep.py` (float and mixed lanes). Runs Metal Performance
Primitives `matmul2d` directly on device-memory tensors (no MLX tiling, no
staging) through stock MLX's `mx.fast.metal_kernel`, for several operand type
combinations and threadgroup tile shapes, at prefill-like GEMM shapes. Every
lane is checked against NumPy before it is timed.

Rate is 2*M*K*N / wall time per call (multiply + add counted separately), so
integer lanes are reported in the same unit as the float sweep. See METHODS.md.

Requirements: an Apple Silicon Mac, MLX + NumPy, and an Xcode/SDK whose
MetalPerformancePrimitives headers provide the tensor ops (Xcode 27 SDK; see
METHODS.md for the toolchain gate).

Usage: python integer_lanes.py [--json out.json] [--reps 7] [--calls 4]
"""


import argparse
import json
import statistics
import time

import numpy as np

import mlx.core as mx

# Wired GPU memory starves macOS if uncapped; cap it.
mx.set_memory_limit(12 * 2**30)
mx.set_cache_limit(2 * 2**30)

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


def pack_nibbles(w):
    u = (w.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def operands(lane, m, k, n, rng, small_ints):
    act, _, wt, _, _, _ = LANES[lane]
    lim = 8 if (small_ints or "int4b" in wt) else 127
    if act == "half":
        a = rng.standard_normal((m, k)).astype(np.float16)
        a_mx = mx.array(a)
    elif act == "bfloat":
        a = rng.standard_normal((m, k)).astype(np.float32)
        a_mx = mx.array(a).astype(mx.bfloat16)
    else:
        a = rng.integers(-lim, lim, size=(m, k)).astype(np.int8)
        a_mx = mx.array(a)
    if "int4b" in wt:
        w = rng.integers(-8, 8, size=(k, n)).astype(np.int8)
        b_mx = mx.array(pack_nibbles(w))
    elif wt == "half":
        w = rng.standard_normal((k, n)).astype(np.float16)
        b_mx = mx.array(w)
    elif wt == "bfloat":
        w = rng.standard_normal((k, n)).astype(np.float32)
        b_mx = mx.array(w).astype(mx.bfloat16)
    else:
        w = rng.integers(-lim, lim, size=(k, n)).astype(np.int8)
        b_mx = mx.array(w)
    mx.eval(a_mx, b_mx)
    return a, w, a_mx, b_mx


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


def check(lane, tile, relaxed, rng):
    m, k, n = 128, 256, 256
    a, w, a_mx, b_mx = operands(lane, m, k, n, rng, small_ints=True)
    out = np.array(call(build(lane, tile, m, k, n, relaxed), lane, tile, a_mx, b_mx, m, n))
    ref = a.astype(np.float64) @ w.astype(np.float64)
    if LANES[lane][4] == "int32_t":
        return bool(np.array_equal(out.astype(np.int64), ref.astype(np.int64))), 0.0
    err = float(np.abs(out.astype(np.float64) - ref).max() / np.abs(ref).max())
    return err < (2e-2 if "bf16" in lane or lane.endswith("f16") else 2e-3), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--relaxed", action="store_true")
    ap.add_argument("--shapes", default="2048x4096x4096,4096x4096x4096")
    ap.add_argument("--lanes", default=None, help="comma-separated lane names, in run order")
    ap.add_argument("--tiles", default=None, help="e.g. 64x128x4,128x128x8")
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    lanes = args.lanes.split(",") if args.lanes else list(LANES)
    tiles = ([tuple(int(v) for v in t.split("x")) for t in args.tiles.split(",")]
             if args.tiles else TILES)

    rng = np.random.default_rng(20261002)
    print(json.dumps(dict(mlx=mx.__version__, device=mx.device_info()), default=str))
    rows = []
    for lane in lanes:
        for tile in tiles:
            try:
                ok, err = check(lane, tile, args.relaxed, rng)
            except Exception as e:
                print(f"{lane:16} tile {tile}: BUILD/RUN FAILED: {str(e)[:160]}")
                rows.append(dict(lane=lane, tile=tile, status="failed", error=str(e)[:400]))
                continue
            if not ok:
                print(f"{lane:16} tile {tile}: WRONG RESULT (rel err {err:.3g})")
                rows.append(dict(lane=lane, tile=tile, status="wrong", rel_err=err))
                continue
            for m, k, n in shapes:
                _, _, a_mx, b_mx = operands(lane, m, k, n, rng, small_ints=False)
                kern = build(lane, tile, m, k, n, args.relaxed)
                fn = lambda: [call(kern, lane, tile, a_mx, b_mx, m, n) for _ in range(args.calls)]
                for _ in range(2):
                    mx.eval(*fn())
                ts = []
                for _ in range(args.reps):
                    outs = fn()
                    t0 = time.perf_counter()
                    mx.eval(*outs)
                    ts.append((time.perf_counter() - t0) / args.calls)
                    del outs
                best, med = min(ts), statistics.median(ts)
                g = lambda t: 2.0 * m * k * n / t / 1e9
                rows.append(dict(lane=lane, tile=tile, status="ok", rel_err=err, M=m, K=k, N=n,
                                 ms_best=best * 1e3, ms_med=med * 1e3,
                                 gops_best=g(best), gops_med=g(med)))
                print(f"{lane:16} tile m{tile[0]}n{tile[1]}s{tile[2]}  {m}x{k}x{n}  "
                      f"best {g(best):8.0f}  median {g(med):8.0f} G(FL)OP/s  err {err:.1e}")
                del a_mx, b_mx
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(mlx=mx.__version__, device=mx.device_info(), relaxed=args.relaxed,
                           calls=args.calls, reps=args.reps, rows=rows), f, indent=1, default=str)


if __name__ == "__main__":
    main()
