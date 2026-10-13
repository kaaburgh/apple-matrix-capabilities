#!/usr/bin/env python3
"""Geometry sweep: prepared-lane rates across a fixed list of real matrix shapes.

Companion to `lane_sweep.py` (same lanes, checks and timing procedure) and to
`integer_lanes.py`. `lane_sweep.py` answers "which lanes does this GPU have, and
how fast at the standard shape". This probe answers the second question a
dispatch rule needs: "how does each lane hold up across the geometries that
real models actually use" - a 2048-token prefill chunk through wide-N, deep-K,
low-rank and narrow families.

It imports `lane_sweep.py` from the same directory and reuses its lane table,
operand encoding, correctness checks, seed and timing loop. It measures a
subset of lanes by default (fp16 baseline, plain fp8, scaled fp8, e5m2); any
lane in `lane_sweep.py`'s table can be selected with --lanes, including packed
fp4 lanes (nibble order is discovered per lane, `lo` first, `hi` if needed).

Correctness: every timed (lane, shape, tile) is validated against the
decoded-operand NumPy reference before timing, as in `lane_sweep.py`. A lane
whose check fails at a shape is recorded `wrong` for that shape and not timed.
Receipts record the probe and helper source hashes (`sources`), toolchain,
power state, the inherited protocol (seed, tiles, warmups, reps), per-lane
precision/packing and per-shape per-tile rates, checks and samples; the
receipt is rewritten atomically after every lane. The exit status is nonzero
when no usable measurement exists. Burst-min rates are published with their
per-tile samples; spread is surfaced by the matrix generator, and unstable
shapes should be rerun before any dispatch rule uses them.

Requirements: an Apple Silicon Mac, MLX + NumPy, and the Xcode 27 SDK (see
METHODS.md for the toolchain gate).

Usage: python geometry_sweep.py [--json geometry_sweep.json] [--force]
       [--lanes lane1,lane2] [--shapes 2048x4096x12288,8192x16384x4096]
"""

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import lane_sweep as ls  # noqa: E402

DEFAULT_LANES = ["f16*f16->f32", "e4m3*e4m3->f32", "mx_e4m3*e4m3->f32", "e5m2*e5m2->f32"]

# The geometry families real prefill traffic is made of (M, K, N). Labels are for
# the receipt; joins against stock-q4 headroom receipts use the tuple. M is the
# 2048-token chunk unless the label says otherwise.
DEFAULT_SHAPES = [
    ("wide_up_4096", (2048, 4096, 12288)),
    ("deep_o_16384", (2048, 16384, 4096)),
    ("deep_down_12288", (2048, 12288, 4096)),
    ("q_fn", (2048, 2560, 12288)),
    ("q_27b", (2048, 5120, 6144)),
    ("up_27b", (2048, 5120, 17408)),
    ("down_27b", (2048, 17408, 5120)),
    ("mla_qb_1024", (2048, 1024, 32768)),
    ("mla_qb_1536", (2048, 1536, 16384)),
    ("mla_kvb_512", (2048, 512, 32768)),
    ("kv_27b", (2048, 5120, 1024)),
    ("kv_step5", (2048, 4096, 768)),
    ("shared_up_fn", (2048, 2560, 640)),
    ("shared_down_fn", (2048, 640, 2560)),
    ("hc_up", (2048, 320, 10240)),
    ("hc_down", (2048, 10240, 320)),
    ("deep_o_16384_m512", (512, 16384, 4096)),
    ("deep_o_16384_m8192", (8192, 16384, 4096)),
    ("deep_down_m512", (512, 12288, 4096)),
    ("deep_down_m8192", (8192, 12288, 4096)),
]

REPS = 5


def tiles_for(m, k, n):
    return [t for t in ls.TILES if m % t[0] == 0 and n % t[1] == 0 and k % 32 == 0]


def is_packed(lane):
    act, _, wt, _, _, _, _ = ls.LANES[lane]
    return "fp4" in act or "fp4" in wt


def discover_packing(lane, shape, rng):
    """Nibble order for packed lanes: try `lo`, then `hi`.

    `lane_sweep.check_tile()` returns `(ok, check_entry, context)`; the check
    entry is only inspected through the tuple. Raises `_common.PackingFailed`
    carrying the orders actually attempted.
    """
    tile = tiles_for(*shape)[0]
    attempts = []
    try:
        attempts.append("lo")
        ok, _chk, _ctx = ls.check_tile(lane, tile, shape, rng, "lo")
        if ok:
            return "lo", attempts
        attempts.append("hi")
        ok, _chk, _ctx = ls.check_tile(lane, tile, shape, rng, "hi")
        if ok:
            return "hi", attempts
        raise C.PackingFailed(attempts)
    except C.PackingFailed:
        raise
    except Exception as exc:
        raise C.PackingFailed(attempts, exc) from exc


def run_lane_shape(lane, label, shape, rng, out, order):
    m, k, n = shape
    entry = {"checks": [], "tile_errors": {}}
    out[lane][label] = entry
    try:
        rates, samples = {}, {}
        for tile in tiles_for(m, k, n):
            tname = f"{tile[0]}x{tile[1]}x{tile[2]}"
            try:
                ok, chk, ctx = ls.check_tile(lane, tile, shape, rng, order)
                entry["checks"].append(C.finite_check(chk))
                if not ok:
                    entry["status"] = "wrong"
                    return
                ts = ls.time_tile(ctx, tile, REPS, 2)
                if min(ts) <= 0 or not _finite(min(ts)):
                    raise ValueError(f"non-finite or non-positive timing sample {min(ts)!r}")
                rates[tname] = round(2.0 * m * k * n / min(ts) / 1e6, 1)
                samples[tname] = [round(t, 3) for t in ts]
            except Exception as exc:
                print(f"[local] {lane} {label} {tname}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                entry["tile_errors"][tname] = C.err_obj(exc, "timing")
        if not rates:
            entry["status"] = "failed"
            entry["error"] = {"phase": "timing", "type": "NoTimedTile",
                              "category": "no_timed_tile"}
            return
        entry.update({"status": "ok", "rates": rates, "samples": samples,
                      "gops_best": max(rates.values())})
    except Exception as exc:
        print(f"[local] {lane} {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
        entry["status"] = "failed"
        entry["error"] = C.err_obj(exc, "build")


def _finite(x):
    import math
    return math.isfinite(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="geometry_sweep.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lanes", default=None, help="comma-separated lane names")
    ap.add_argument("--shapes", default=None, help="m,k,n tuples separated by commas")
    args = ap.parse_args()
    lanes = args.lanes.split(",") if args.lanes else DEFAULT_LANES
    for lane in lanes:
        if lane not in ls.LANES:
            print(f"unknown lane: {lane}", file=sys.stderr)
            return 2
    try:
        if args.shapes:
            shapes = [(f"shape_{i}", C.parse_int_tuple(s, "--shapes"))
                      for i, s in enumerate(args.shapes.split(","))]
        else:
            shapes = [(label, C.parse_int_tuple("x".join(map(str, s)), label))
                      for label, s in DEFAULT_SHAPES]
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2
    for label, shape in shapes:
        if not tiles_for(*shape):
            print(f"argument error: shape {label} {shape} has no eligible tile "
                  f"(needs M/N divisible by a listed tile and K % 32 == 0)",
                  file=sys.stderr)
            return 2
    if os.path.exists(args.json) and not args.force:
        print(f"{args.json} exists; pass --force to overwrite "
              "(receipts are evidence - prefer a new name)", file=sys.stderr)
        return 2

    out = ls.machine_meta()
    out["probe"] = "apple-matrix-capabilities/geometry_sweep"
    out["probe_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    out["sources"] = C.source_hashes(C.probe_files(__file__, ls.__file__))
    out["protocol"] = {
        "shapes": {label: list(s) for label, s in shapes},
        "lanes_requested": lanes,
        "tiles": [f"{t[0]}x{t[1]}x{t[2]}" for t in ls.TILES],
        "check_shape": list(ls.CHK),
        "rate_shape": list(ls.RATE),
        "warmups": 2,
        "reps": REPS,
        "seed": ls.SEED,
        "mode": "multiply",
        "sample": "one kernel call per timed sample, call construction inside the "
                  "timer; best-of-reps per tile, best-of-tiles per lane/shape",
    }
    out["lane_meta"] = {lane: {"relaxed_precision": ls.lane_relaxed(lane),
                               "packing": None, "packing_attempts": []}
                        for lane in lanes}
    out["lanes"] = {}
    out["complete"] = False
    write_receipt = C.write_receipt
    write_receipt(out, args.json)
    for lane in lanes:
        out["lanes"][lane] = {}
        rng = np.random.default_rng([ls.SEED, ls.LANE_INDEX[lane]])
        if is_packed(lane):
            shape = next(s for _, s in shapes if tiles_for(*s))
            try:
                order, attempts = discover_packing(lane, shape, rng)
            except Exception as exc:
                attempts = list(getattr(exc, "attempts", []))
                out["lane_meta"][lane]["packing_attempts"] = attempts
                out["lane_meta"][lane]["packing_error"] = {
                    "type": type(exc).__name__, "category": C.classify(
                        getattr(exc, "cause", None) or exc)}
                print(f"[local] {lane}: {type(exc).__name__}: {exc}", file=sys.stderr)
                for label, _ in shapes:
                    out["lanes"][lane][label] = {
                        "status": "failed", "checks": [], "tile_errors": {},
                        "error": C.err_obj(exc, "packing_check")}
                print(f"{lane:18} PACKING CHECK FAILED {type(exc).__name__}",
                      flush=True)
                write_receipt(out, args.json)
                continue
            out["lane_meta"][lane]["packing"] = order
            out["lane_meta"][lane]["packing_attempts"] = attempts
        else:
            order = "lo"
        for label, shape in shapes:
            run_lane_shape(lane, label, shape, rng, out["lanes"], order)
            e = out["lanes"][lane][label]
            if e["status"] == "ok":
                print(f"{lane:18} {label:18} {e['gops_best']:9,.1f} GOPS", flush=True)
            else:
                print(f"{lane:18} {label:18} {e['status'].upper()}", flush=True)
            write_receipt(out, args.json)
    ok = sum(1 for lane in lanes for e in out["lanes"][lane].values()
             if e.get("status") == "ok")
    out["results"] = {"cells_ok": ok,
                      "cells_total": sum(len(v) for v in out["lanes"].values())}
    out["complete"] = True
    write_receipt(out, args.json)
    print(f"receipt: {args.json} ({out['results']})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
