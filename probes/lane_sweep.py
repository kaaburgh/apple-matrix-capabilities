#!/usr/bin/env python3
"""Lane sweep: measured matrix-unit lane rates on an Apple Silicon GPU.

Runs a fixed set of operand-format lanes through `mpp::tensor_ops::matmul2d`
fed from device memory and writes a receipt `lane_sweep.json` next to this
script. See METHODS.md for the protocol and its traps, and README.md for how
the receipt feeds the capability matrix.

Requirements: an Apple Silicon Mac, a Python with MLX (pip) and NumPy, and an
Xcode/SDK whose MetalPerformancePrimitives headers provide the tensor ops.
Currently that means the Xcode 27 SDK: with Xcode 26.6 MLX (0.29-0.32) fails
to JIT-compile even fp16 lanes, because its injected kernel headers reference
fp8 `metal` types the older Metal standard library lacks. See METHODS.md.
Unsupported lanes fail per lane and are recorded as failed, not aborted.

Lanes:
  e4m3        untransposed unscaled fp8
  t_e4m3      transposed unscaled fp8   (layout cost)
  mx_e4m3     transposed + UE8M0 scale plane
  mx_fp4      fp4 x fp4 + UE8M0 plane
  f16*f16     fp16 x fp16 (reference lane)
  fp4*fp4     fp4 x fp4
  f16*fp4     f16 x fp4 (unscaled mixed)
  f16*e4m3    f16 x e4m3 (unscaled mixed)
  f16_mx_e4m3 f16 x e4m3 + UE8M0 plane (weights-only staging shape)

Correctness via constant encodings (1.0 x 2.0), plus the per-n/per-kb block-scale
check on scaled lanes. mode::multiply only.
"""

import json
import os
import time

import numpy as np

import mlx.core as mx

mx.set_memory_limit(12 * 2**30)
mx.set_cache_limit(2 * 2**30)

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
using namespace mpp::tensor_ops;
"""

SRC = """
    constexpr auto desc = matmul2d_descriptor(
        {mt}, {nt}, static_cast<int>(metal::dynamic_extent), false, {tr}, {rel},
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, metal::execution_simdgroups<{sg}>> op;
    auto tgx = (int)threadgroup_position_in_grid.x;
    auto tgy = (int)threadgroup_position_in_grid.y;
{defs}
    auto mA = tA.slice(0, tgy * {mt});
    {bslice}
    auto mC = tC.slice(tgx * {nt}, tgy * {mt});
    op.run(mA, mB, mC);
"""

TAIL_UNTRANS = """    auto tC = metal::tensor<device {acc}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {acc}*)C, metal::dextents<int32_t, 2>({N}, {M}));"""

DEFS_UNTRANS = """    auto tA = metal::tensor<device {act}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {acast}*)A, metal::dextents<int32_t, 2>({K}, {M}));
    auto tB = metal::tensor<device {wt}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {wcast}*)B, metal::dextents<int32_t, 2>({N}, {K}));
""" + TAIL_UNTRANS

DEFS_TRANS = """    auto tA = metal::tensor<device {act}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {acast}*)A, metal::dextents<int32_t, 2>({K}, {M}), metal::array<int32_t, 2>{{1, {K}}});
    auto tB = metal::tensor<device {wt}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {wcast}*)B, metal::dextents<int32_t, 2>({K}, {N}), metal::array<int32_t, 2>{{1, {K}}});
""" + TAIL_UNTRANS

DEFS_SCALED = """    using scales_tag = metal::tensor_blockwise<metal::tensor_plane_scales, device metal::metal_fp8_ue8m0_format, 32, 1>;
    scales_tag planeB((scales_tag::data_handle_type)SB);
    auto tA = metal::tensor<device {act}, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        (device {acast}*)A, metal::dextents<int32_t, 2>({K}, {M}), metal::array<int32_t, 2>{{1, {K}}});
    auto tB = metal::tensor<device {wt}, metal::dextents<int32_t, 2>, metal::tensor_inline, scales_tag>(
        (device {wcast}*)B, metal::dextents<int32_t, 2>({K}, {N}), metal::array<int32_t, 2>{{1, {K}}}, planeB);
""" + TAIL_UNTRANS

# name: act, acast, wt, wcast, acc, scaled, transposed
LANES = {
    "e4m3*e4m3->f32": ("metal::metal_fp8_e4m3_format", "uchar", "metal::metal_fp8_e4m3_format", "uchar", "float", False, False),
    "t_e4m3*e4m3->f32": ("metal::metal_fp8_e4m3_format", "uchar", "metal::metal_fp8_e4m3_format", "uchar", "float", False, True),
    "mx_e4m3*e4m3->f32": ("metal::metal_fp8_e4m3_format", "uchar", "metal::metal_fp8_e4m3_format", "uchar", "float", True, True),
    "mx_fp4*fp4->f32": ("metal::metal_fp4_e2m1_format", "uchar", "metal::metal_fp4_e2m1_format", "uchar", "float", True, True),
    "f16_mx_fp4->f32": ("half", "half", "metal::metal_fp4_e2m1_format", "uchar", "float", True, True),
    "f16*f16->f32": ("half", "half", "half", "half", "float", False, False),
    "fp4*fp4->f32": ("metal::metal_fp4_e2m1_format", "uchar", "metal::metal_fp4_e2m1_format", "uchar", "float", False, False),
    "f16*fp4->f32": ("half", "half", "metal::metal_fp4_e2m1_format", "uchar", "float", False, False),
    "f16*e4m3->f32": ("half", "half", "metal::metal_fp8_e4m3_format", "uchar", "float", False, False),
    "f16_mx_e4m3->f32": ("half", "half", "metal::metal_fp8_e4m3_format", "uchar", "float", True, True),
}
TILES = [(64, 128, 4), (64, 64, 2), (64, 64, 4), (64, 32, 4), (128, 128, 8)]
CHK = (64, 1024, 256)
RATE = (2048, 4096, 4096)


def vt_bytes(v, kind):
    if kind == "fp4":
        nib = {1.0: 0x2, 2.0: 0x4}[v]
        return (nib << 4) | nib
    if kind == "fp8":
        return {1.0: 0x38, 2.0: 0x40}[v]
    return np.float16(v)


def vkind(t):
    return "fp4" if "fp4" in t else ("fp8" if "fp8" in t else "f16")


def operand(lane, val, m, n, k, which):
    act, _, wt, _, _, scaled, trans = LANES[lane]
    t = act if which == "A" else wt
    kind = vkind(t)
    fast = "fp4" in t
    if which == "A":
        shape = (m, k // 2) if fast else (m, k)
    else:
        if scaled or trans:
            shape = (n, k // 2) if fast else (n, k)
        else:
            shape = (k, n // 2) if fast else (k, n)
    if kind == "f16":
        return mx.array(np.full(shape, val, dtype=np.float16))
    return mx.array(np.full(shape, vt_bytes(val, kind), dtype=np.uint8))


def build(lane, m, k, n, tile):
    mt, nt, sg = tile
    act, acast, wt, wcast, acc, scaled, trans = LANES[lane]
    ak = 2 if "fp4" in act else 1
    wk = 2 if "fp4" in wt else 1
    tmpl = DEFS_SCALED if scaled else (DEFS_TRANS if trans else DEFS_UNTRANS)
    defs = tmpl.format(act=act, acast=acast, wt=wt, wcast=wcast, acc=acc, K=k, M=m, N=n,
                       akstr=f"{k} / {ak}" if ak == 2 else str(k),
                       wkstr=f"{k} / {wk}" if wk == 2 else str(k))
    tag = "".join(c if c.isalnum() else "_" for c in lane)
    src = SRC.format(mt=mt, nt=nt, sg=sg, tr=("true" if (scaled or trans) else "false"),
                     rel=("false" if scaled else "true"),
                     bslice=("auto mB = tB.slice(0, tgx * " + f"{nt});" if (scaled or trans)
                             else f"auto mB = tB.slice(tgx * {nt}, 0);"),
                     defs=defs, M=m, K=k, N=n)
    names = ["A", "B", "SB"] if scaled else ["A", "B"]
    return mx.fast.metal_kernel(name=f"amc_{tag}_{m}x{k}x{n}_{mt}_{nt}_{sg}",
                                input_names=names, output_names=["C"], header=HEADER, source=src)


def call(kern, inputs, m, n, tile):
    mt, nt, sg = tile
    return kern(inputs=inputs, grid=(sg * 32 * (n // nt), m // mt, 1),
                threadgroup=(sg * 32, 1, 1), output_shapes=[(m, n)],
                output_dtypes=[mx.float32])[0]


def run_lane(lane, out):
    act, _, wt, _, _, scaled, trans = LANES[lane]
    m, k, n = CHK
    tile = TILES[0]
    try:
        a = operand(lane, 1.0, m, n, k, "A")
        b = operand(lane, 2.0, m, n, k, "B")
        inputs = [a, b]
        if scaled:
            inputs.append(mx.array(np.full((n, k // 32), 0x7F, dtype=np.uint8)))
        kern = build(lane, m, k, n, tile)
        y = np.array(call(kern, inputs, m, n, tile))
        err = float(np.abs(y - 2.0 * k).max() / (2.0 * k))
        ok = err < 1e-3
        diag = ""
        if not ok:
            diag = f" y00={y[0, 0]} min={y.min()} max={y.max()}"
        if scaled:
            sb = np.full((n, k // 32), 0x7F, dtype=np.uint8)
            sb[:, 1::2] = 0x80
            sb[1::2, :] = 0x80
            inputs[-1] = mx.array(sb)
            y2 = np.array(call(kern, inputs, m, n, tile))
            e_e = float(np.abs(y2[:, 0::2] - 3.0 * k).max() / (3.0 * k))
            e_o = float(np.abs(y2[:, 1::2] - 4.0 * k).max() / (4.0 * k))
            ok = ok and max(e_e, e_o) < 1e-3
            diag += f" mx even={e_e:.1e} odd={e_o:.1e}"
    except Exception as exc:
        print(f"{lane:18} FAILED: {str(exc)[:150]}", flush=True)
        out[lane] = {"status": "failed", "error": str(exc)[:300]}
        return
    m2, k2, n2 = RATE
    rates = {}
    for t in TILES:
        try:
            a2 = operand(lane, 1.0, m2, n2, k2, "A")
            b2 = operand(lane, 2.0, m2, n2, k2, "B")
            inputs2 = [a2, b2]
            if scaled:
                inputs2.append(mx.array(np.full((n2, k2 // 32), 0x7F, dtype=np.uint8)))
            kern2 = build(lane, m2, k2, n2, t)
            for _ in range(2):
                mx.eval(call(kern2, inputs2, m2, n2, t))
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                mx.eval(call(kern2, inputs2, m2, n2, t))
                ts.append(time.perf_counter() - t0)
            rates[t] = 2.0 * m2 * k2 * n2 / min(ts) / 1e9
        except Exception as exc:
            rates[t] = -1.0
            diag += f" rf[{t}]:{str(exc)[:50]}"
    best = max(rates.values())
    out[lane] = {"status": "ok" if ok else "wrong", "diag": diag, "gops_best": best,
                 "rates": {f"{t[0]}x{t[1]}x{t[2]}": round(v, 1) for t, v in rates.items()}}
    print(f"{lane:18} {'ok' if ok else 'WRONG'} best {best:8.0f} GOPS "
          f"tiles {{ {', '.join(f'{t[0]}x{t[1]}x{t[2]}:{v:.0f}' for t, v in rates.items())} }}{diag}", flush=True)


def main():
    out = {"probe": "apple-matrix-capabilities/lane_sweep", "mlx": mx.__version__, "device": mx.device_info()}
    for lane in LANES:
        run_lane(lane, out)
    ref = out.get("e4m3*e4m3->f32", {}).get("gops_best", 0)
    if ref:
        for lane, r in out.items():
            if isinstance(r, dict) and r.get("gops_best", 0) > 0:
                r["vs_e4m3"] = round(r["gops_best"] / ref, 3)
                print(f"{lane:18} vs e4m3: {r['vs_e4m3']:.2f}x", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "lane_sweep.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
