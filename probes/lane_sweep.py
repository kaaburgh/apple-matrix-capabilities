#!/usr/bin/env python3
"""Lane sweep: measured matrix-unit lane rates on an Apple Silicon GPU.

Runs a fixed set of operand-format lanes through `mpp::tensor_ops::matmul2d`
fed from device memory and writes a receipt JSON. See METHODS.md for the
protocol and its traps, README.md for how receipts feed the capability matrix.

Correctness (receipt schema 2): every timed tile is validated at the timing
shape against a decoded-operand NumPy reference built from deterministic,
nonuniform, sign-varying values that are exactly representable in every lane
format (and under any mantissa truncation). Packed lanes (fp4) are validated
in both nibble orders; the matching order is recorded. A lane whose checks
fail is never timed. Failures are recorded per lane/tile with structured,
sanitized error objects, and the receipt is rewritten after every lane.

Requirements: an Apple Silicon Mac, a Python with MLX (pip) and NumPy, and an
Xcode/SDK whose MetalPerformancePrimitives headers provide the tensor ops.
Currently that means the Xcode 27 SDK: with Xcode 26.6 MLX (0.29-0.32) fails
to JIT-compile even fp16 lanes, because its injected kernel headers reference
fp8 `metal` types the older Metal standard library lacks. See METHODS.md.

Lanes:
  e4m3        untransposed unscaled fp8
  t_e4m3      transposed unscaled fp8   (layout cost)
  mx_e4m3     transposed + UE8M0 scale plane
  mx_fp4      fp4 x fp4 + UE8M0 plane
  f16_mx_fp4  f16 x fp4 + UE8M0 plane
  f16*f16     fp16 x fp16 (reference lane)
  fp4*fp4     fp4 x fp4
  f16*fp4     f16 x fp4 (unscaled mixed)
  f16*e4m3    f16 x e4m3 (unscaled mixed)
  f16_mx_e4m3 f16 x e4m3 + UE8M0 plane (weights-only staging shape)
  f32*f32     fp32 x fp32 (relaxed, like the other unscaled lanes)
  strict_f32  fp32 x fp32 with relaxed_precision off
  strict_f16  fp16 x fp16 with relaxed_precision off
  strict_e4m3 untransposed fp8 with relaxed_precision off
  relaxed_mx_e4m3  the scaled fp8 lane with relaxed_precision on
                   (separates the scale-plane cost from the relaxed flag)
  e5m2        fp8 e5m2 (the other 8-bit float: wider range, 2 mantissa bits)
  strict_e5m2 e5m2 with relaxed_precision off
  t_e5m2      transposed e5m2
  f16*e5m2    f16 x e5m2 (unscaled mixed)
  mx_e5m2     e5m2 x e5m2 + UE8M0 plane

Unscaled lanes run with relaxed_precision on (as in the original sweep) and
scaled lanes off; `strict_` and `relaxed_` mark the exceptions. Every receipt
records the flag per lane. mode::multiply only.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np

import mlx.core as mx

RECEIPT_SCHEMA = 2
SEED = 20261012
TOL = 1e-5


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
    "f32*f32->f32": ("float", "float", "float", "float", "float", False, False),
    "strict_f32*f32->f32": ("float", "float", "float", "float", "float", False, False),
    "strict_f16*f16->f32": ("half", "half", "half", "half", "float", False, False),
    "strict_e4m3*e4m3->f32": ("metal::metal_fp8_e4m3_format", "uchar", "metal::metal_fp8_e4m3_format", "uchar", "float", False, False),
    "relaxed_mx_e4m3*e4m3->f32": ("metal::metal_fp8_e4m3_format", "uchar", "metal::metal_fp8_e4m3_format", "uchar", "float", True, True),
    "e5m2*e5m2->f32": ("metal::metal_fp8_e5m2_format", "uchar", "metal::metal_fp8_e5m2_format", "uchar", "float", False, False),
    "strict_e5m2*e5m2->f32": ("metal::metal_fp8_e5m2_format", "uchar", "metal::metal_fp8_e5m2_format", "uchar", "float", False, False),
    "t_e5m2*e5m2->f32": ("metal::metal_fp8_e5m2_format", "uchar", "metal::metal_fp8_e5m2_format", "uchar", "float", False, True),
    "f16*e5m2->f32": ("half", "half", "metal::metal_fp8_e5m2_format", "uchar", "float", False, False),
    "mx_e5m2*e5m2->f32": ("metal::metal_fp8_e5m2_format", "uchar", "metal::metal_fp8_e5m2_format", "uchar", "float", True, True),
}
# relaxed_precision per lane: unscaled lanes default on, scaled lanes default
# off; these are the explicit exceptions. `strict_` marks relaxed off, `relaxed_`
# marks a scaled lane with relaxed on.
RELAXED = {"strict_f32*f32->f32": False, "strict_f16*f16->f32": False,
           "strict_e4m3*e4m3->f32": False, "relaxed_mx_e4m3*e4m3->f32": True,
           "strict_e5m2*e5m2->f32": False}


def lane_relaxed(lane):
    return RELAXED.get(lane, not LANES[lane][5])

TILES = [(64, 128, 4), (64, 64, 2), (64, 64, 4), (64, 32, 4), (128, 128, 8)]
CHK = (64, 1024, 256)
RATE = (2048, 4096, 4096)

# Nonuniform, sign-varying, exactly representable in fp4 e2m1, fp8 e4m3, fp16,
# bf16 and under any mantissa truncation of fp32 (<= 1 significant mantissa bit).
VALUES = np.array([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], dtype=np.float32)
FP4_BITS = np.array([0x1, 0x9, 0x2, 0xA, 0x4, 0xC, 0x5, 0xD, 0x6, 0xE], dtype=np.uint8)
FP8_BITS = np.array([0x30, 0xB0, 0x38, 0xB8, 0x40, 0xC0, 0x44, 0xC4, 0x48, 0xC8], dtype=np.uint8)
FP8_E5M2_BITS = np.array([0x38, 0xB8, 0x3C, 0xBC, 0x40, 0xC0, 0x42, 0xC2, 0x44, 0xC4], dtype=np.uint8)
SCALE_VALUES = np.array([0.5, 1.0, 2.0], dtype=np.float32)
SCALE_BITS = np.array([0x7E, 0x7F, 0x80], dtype=np.uint8)


def lane_kind(t):
    if "fp4" in t:
        return "fp4"
    if "e5m2" in t:
        return "e5m2"
    if "fp8" in t:
        return "fp8"
    return "f32" if t == "float" else "f16"


def pattern(rng, shape, kind):
    idx = rng.integers(0, len(VALUES), size=shape)
    if kind == "fp4":
        return VALUES[idx], FP4_BITS[idx]
    if kind == "fp8":
        return VALUES[idx], FP8_BITS[idx]
    if kind == "e5m2":
        return VALUES[idx], FP8_E5M2_BITS[idx]
    return VALUES[idx], None


def pack(bits, axis, order):
    lo = np.take(bits, np.arange(0, bits.shape[axis], 2), axis=axis)
    hi = np.take(bits, np.arange(1, bits.shape[axis], 2), axis=axis)
    return (lo | (hi << 4)).astype(np.uint8) if order == "lo" else (hi | (lo << 4)).astype(np.uint8)


def encoded_operands(lane, a_val, a_bits, b_val, b_bits, s_val, order):
    """Layouts mirror lane_sweep v1: A packs along K; B packs along K when
    transposed/scaled, along N otherwise."""
    act, _, wt, _, _, scaled, trans = LANES[lane]
    ka, kw = lane_kind(act), lane_kind(wt)
    if ka == "f16":
        a_mx = mx.array(a_val.astype(np.float16))
    elif ka == "f32":
        a_mx = mx.array(a_val)
    elif ka == "fp4":
        a_mx = mx.array(pack(a_bits, 1, order))
    else:  # fp8: one byte per element, no packing
        a_mx = mx.array(a_bits)
    if kw == "f16":
        b_mx = mx.array((b_val.T if (scaled or trans) else b_val).astype(np.float16))
    elif kw == "f32":
        b_mx = mx.array(b_val.T if (scaled or trans) else b_val)
    elif kw == "fp8" or kw == "e5m2":
        b_mx = mx.array((b_bits.T if (scaled or trans) else b_bits))
    else:
        b_mx = mx.array(pack(b_bits, 0, order).T if (scaled or trans)
                        else pack(b_bits, 1, order))
    inputs = [a_mx, b_mx]
    if scaled:
        sb = SCALE_BITS[np.searchsorted(SCALE_VALUES, s_val)]
        inputs.append(mx.array(sb.astype(np.uint8)))
    return inputs, a_val, b_val


def reference(a_val, b_val, s_val, k):
    b = b_val.astype(np.float32)
    if s_val is not None:
        b = b * np.repeat(s_val.T, 32, axis=0)
    return a_val @ b


def build(lane, m, k, n, tile):
    mt, nt, sg = tile
    act, acast, wt, wcast, acc, scaled, trans = LANES[lane]
    tmpl = DEFS_SCALED if scaled else (DEFS_TRANS if trans else DEFS_UNTRANS)
    defs = tmpl.format(act=act, acast=acast, wt=wt, wcast=wcast, acc=acc, K=k, M=m, N=n)
    tag = "".join(c if c.isalnum() else "_" for c in lane)
    src = SRC.format(mt=mt, nt=nt, sg=sg, tr=("true" if (scaled or trans) else "false"),
                     rel=("true" if lane_relaxed(lane) else "false"),
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


def rel_err(y, ref):
    return float(np.abs(y - ref).max() / max(float(np.abs(ref).max()), 1.0))


def check_tile(lane, tile, shape, rng, order):
    """Build, run and validate one tile at one shape. Returns (ok, entry, ctx)."""
    m, k, n = shape
    act, _, wt, _, _, scaled, _ = LANES[lane]
    a_val, a_bits = pattern(rng, (m, k), lane_kind(act))
    b_val, b_bits = pattern(rng, (k, n), lane_kind(wt))
    s_val = (SCALE_VALUES[rng.integers(0, len(SCALE_VALUES), size=(n, k // 32))]
             if scaled else None)
    inputs, av, bv = encoded_operands(lane, a_val, a_bits, b_val, b_bits, s_val, order)
    kern = build(lane, m, k, n, tile)
    y = np.array(call(kern, inputs, m, n, tile))
    ref = reference(av, bv, s_val, k)
    err = rel_err(y, ref)
    entry = {"stage": f"{'check' if shape == CHK else 'rate'}_shape",
             "tile": f"{tile[0]}x{tile[1]}x{tile[2]}", "rel_err": err,
             "tolerance": TOL, "ok": bool(err < TOL)}
    ctx = {"kern": kern, "inputs": inputs, "m": m, "n": n}
    return entry["ok"], entry, ctx


def time_tile(ctx, tile, reps, warmups):
    m, n = ctx["m"], ctx["n"]

    def one():
        x = ctx["kern"](inputs=ctx["inputs"], grid=(tile[2] * 32 * (n // tile[1]), m // tile[0], 1),
                        threadgroup=(tile[2] * 32, 1, 1), output_shapes=[(m, n)],
                        output_dtypes=[mx.float32])[0]
        mx.eval(x)

    for _ in range(warmups):
        one()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        one()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def run_lane(lane, out, rng, reps, warmups):
    entry = {"status": "failed", "relaxed_precision": lane_relaxed(lane),
             "packing": None, "checks": [], "tile_errors": {}}
    out[lane] = entry
    try:
        order = "lo"
        ok0, chk0, _ = check_tile(lane, TILES[0], CHK, rng, order)
        if not ok0 and (lane_kind(LANES[lane][0]) == "fp4" or lane_kind(LANES[lane][2]) == "fp4"):
            if lane_kind(LANES[lane][0]) == "fp4":
                ok0b, chk0b, _ = check_tile(lane, TILES[0], CHK, rng, "hi")
                if ok0b:
                    order, ok0, chk0 = "hi", ok0b, chk0b
            elif lane_kind(LANES[lane][2]) == "fp4":
                ok0b, chk0b, _ = check_tile(lane, TILES[0], CHK, rng, "hi")
                if ok0b:
                    order, ok0, chk0 = "hi", ok0b, chk0b
        entry["packing"] = order if (lane_kind(LANES[lane][0]) == "fp4"
                                     or lane_kind(LANES[lane][2]) == "fp4") else None
        entry["checks"].append(chk0)
        if not ok0:
            entry["status"] = "wrong"
            return
        rates, samples = {}, {}
        for tile in TILES:
            tname = f"{tile[0]}x{tile[1]}x{tile[2]}"
            try:
                ok, chk, ctx = check_tile(lane, tile, RATE, rng, order)
                entry["checks"].append(chk)
                if not ok:
                    entry["status"] = "wrong"
                    return
                ts = time_tile(ctx, tile, reps, warmups)
                rates[tname] = round(2.0 * RATE[0] * RATE[1] * RATE[2] / min(ts) / 1e6, 1)
                samples[tname] = [round(t, 3) for t in ts]
            except Exception as exc:
                entry["tile_errors"][tname] = err_obj(exc, "timing")
        if not rates:
            entry["status"] = "failed"
            entry["error"] = {"phase": "timing", "type": "NoTimedTile", "message": ""}
            return
        entry.update({"status": "ok", "rates": rates, "samples": samples,
                      "gops_best": max(rates.values()), "gops_med": round(statistics.median(
                          [2.0 * RATE[0] * RATE[1] * RATE[2] / (t / 1e3) / 1e9
                           for ts in samples.values() for t in [statistics.median(ts)]]), 1)})
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = err_obj(exc, "build")


def sanitize(text, limit=200):
    t = str(text).splitlines()[0].strip() if text else ""
    t = re.sub(r"(?:/Users|/private|/var/folders|/opt|/Volumes)/[^\s\"']*", "<path>", t)
    return t[:limit]


def err_obj(exc, phase):
    return {"phase": phase, "type": type(exc).__name__, "message": sanitize(exc)}


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        first = r.stdout.strip().splitlines()
        return first[0] if first else "unknown"
    except Exception:
        return "unknown"


def machine_meta():
    dev = mx.device_info()
    power = {"source": "unknown", "low_power_mode": "unknown"}
    try:
        batt = sh(["pmset", "-g", "batt"])
        power["source"] = ("ac" if "AC Power" in batt
                           else "battery" if "Battery Power" in batt else "unknown")
        m = re.search(r"lowpowermode\s+(\d)",
                      subprocess.run(["pmset", "-g"], capture_output=True, text=True,
                                     timeout=5).stdout)
        if m:
            power["low_power_mode"] = (m.group(1) == "1")
    except Exception:
        pass
    try:
        probe_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    except Exception:
        probe_hash = "unknown"
    return {
        "probe": "apple-matrix-capabilities/lane_sweep",
        "receipt_schema": RECEIPT_SCHEMA,
        "probe_sha256": probe_hash,
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": {"version": platform.mac_ver()[0], "build": sh(["sw_vers", "-buildVersion"])},
        "toolchain": {"xcode": sh(["xcodebuild", "-version"]),
                      "sdk": sh(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
                      "metal": sh(["xcrun", "metal", "--version"])},
        "power": power,
        "env": {"MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32", "unset")},
        "device": {k: dev.get(k) for k in ("device_name", "architecture", "memory_size",
                                           "max_recommended_working_set_size")},
    }


def protocol_meta(args):
    return {"shape": list(RATE), "check_shape": list(CHK), "tiles": [
        f"{t[0]}x{t[1]}x{t[2]}" for t in TILES], "warmups": args.warmups, "reps": args.reps,
        "sample": "one kernel call per timed sample, call construction inside the timer; "
                  "best-of-reps per tile, best-of-tiles per lane",
        "mode": "multiply", "seed": args.seed, "lanes_requested": list(LANES) if not args.lanes
        else args.lanes.split(",")}


def summarize(out):
    st = [out[k]["status"] for k in LANES if isinstance(out.get(k), dict)]
    out["results"] = {"ok": st.count("ok"), "wrong": st.count("wrong"),
                      "failed": st.count("failed"), "lanes_total": len(st)}


def write_receipt(out, path):
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="lane_sweep.json",
                    help="receipt path (default lane_sweep.json in the working directory)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing receipt")
    ap.add_argument("--lanes", default=None, help="comma-separated lane names, in run order")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.reps < 1 or args.warmups < 1:
        print("--reps and --warmups must be positive", file=sys.stderr)
        return 2
    lanes = list(LANES) if not args.lanes else args.lanes.split(",")
    for lane in lanes:
        if lane not in LANES:
            print(f"unknown lane: {lane}", file=sys.stderr)
            return 2
    if os.path.exists(args.json) and not args.force:
        print(f"{args.json} exists; pass --force to overwrite "
              "(receipts are evidence - prefer a new name)", file=sys.stderr)
        return 2
    out = machine_meta()
    out["protocol"] = protocol_meta(args)
    out["results"] = {}
    write_receipt(out, args.json)
    for lane in lanes:
        rng = np.random.default_rng([args.seed, list(LANES).index(lane)])
        run_lane(lane, out, rng, args.reps, args.warmups)
        summarize(out)
        entry = out[lane]
        if entry["status"] == "ok":
            print(f"{lane:18} ok best {entry['gops_best']:9,.1f} GOPS "
                  f"tiles {{ {', '.join(f'{k}:{v:.0f}' for k, v in entry['rates'].items())} }} "
                  f"packing={entry['packing']}", flush=True)
        else:
            print(f"{lane:18} {entry['status'].upper()} {entry.get('error', '')} "
                  f"{entry['tile_errors'] or ''}", flush=True)
        write_receipt(out, args.json)
    ref = out.get("e4m3*e4m3->f32", {})
    ref = ref.get("gops_best", 0) if isinstance(ref, dict) else 0
    if ref:
        for lane in lanes:
            r = out[lane]
            if r["status"] == "ok":
                r["vs_e4m3"] = round(r["gops_best"] / ref, 3)
                print(f"{lane:18} vs e4m3: {r['vs_e4m3']:.2f}x", flush=True)
    summarize(out)
    write_receipt(out, args.json)
    print(f"receipt: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
