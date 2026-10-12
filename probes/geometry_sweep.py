#!/usr/bin/env python3
"""Geometry sweep: prepared-lane rates across a fixed list of real matrix shapes.

Companion to `lane_sweep.py` (same lanes, checks and timing procedure) and to
`integer_lanes.py`. `lane_sweep.py` answers "which lanes does this GPU have, and
how fast at the standard shape". This probe answers the second question a
dispatch rule needs: "how does each lane hold up across the geometries that
real models actually use" - a 2048-token prefill chunk through wide-N, deep-K,
low-rank and narrow families.

It imports `lane_sweep.py` from the same directory and reuses its lane table,
operand encoding, correctness checks and timing loop. It measures a subset of
lanes by default (fp16 baseline, plain fp8, scaled fp8, e5m2); packed fp4 lanes
are supported via the same machinery if selected with --lanes.

Correctness: every timed (lane, shape, tile) is validated against the
decoded-operand NumPy reference before timing, as in `lane_sweep.py`. A lane
whose check fails at a shape is recorded `wrong` for that shape and not timed.
Receipts record the probe SHA-256, toolchain, power state, per-shape per-tile
rates, checks and samples; the receipt is rewritten after every lane.

Requirements: an Apple Silicon Mac, MLX + NumPy, and the Xcode 27 SDK (see
METHODS.md for the toolchain gate).

Usage: python geometry_sweep.py [--json geometry_sweep.json] [--force]
       [--lanes lane1,lane2] [--shapes 2048x4096x12288,8192x16384x4096]
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def run_lane_shape(lane, label, shape, rng, out):
    m, k, n = shape
    entry = {"checks": [], "tile_errors": {}}
    out[lane][label] = entry
    try:
        rates, samples = {}, {}
        for tile in tiles_for(m, k, n):
            tname = f"{tile[0]}x{tile[1]}x{tile[2]}"
            try:
                ok, chk, ctx = ls.check_tile(lane, tile, shape, rng, "lo")
                entry["checks"].append(chk)
                if not ok:
                    entry["status"] = "wrong"
                    return
                ts = ls.time_tile(ctx, tile, REPS, 2)
                rates[tname] = round(2.0 * m * k * n / min(ts) / 1e6, 1)
                samples[tname] = [round(t, 3) for t in ts]
            except Exception as exc:
                entry["tile_errors"][tname] = ls.err_obj(exc, "timing")
        if not rates:
            entry["status"] = "failed"
            entry["error"] = {"phase": "timing", "type": "NoTimedTile", "message": ""}
            return
        entry.update({"status": "ok", "rates": rates, "samples": samples,
                      "gops_best": max(rates.values())})
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = ls.err_obj(exc, "build")


def write_receipt(out, path):
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)


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
    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = [(f"shape_{s}", tuple(int(v) for v in s.split("x")))
                  for s in args.shapes.split(",")]
    if os.path.exists(args.json) and not args.force:
        print(f"{args.json} exists; pass --force to overwrite "
              "(receipts are evidence - prefer a new name)", file=sys.stderr)
        return 2
    out = ls.machine_meta()
    out["probe"] = "apple-matrix-capabilities/geometry_sweep"
    out["probe_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    out["protocol"] = {"shapes": {label: list(s) for label, s in shapes},
                       "lanes": lanes, "warmups": 2, "reps": REPS,
                       "rate": "best of reps per tile, best of tiles per lane/shape"}
    out["lanes"] = {}
    for lane in lanes:
        out["lanes"][lane] = {}
        rng = np.random.default_rng([ls.SEED, lanes.index(lane)])
        for label, shape in shapes:
            run_lane_shape(lane, label, shape, rng, out["lanes"])
            e = out["lanes"][lane][label]
            if e["status"] == "ok":
                print(f"{lane:18} {label:18} {e['gops_best']:9,.1f} GOPS", flush=True)
            else:
                print(f"{lane:18} {label:18} {e['status'].upper()}", flush=True)
            write_receipt(out, args.json)
    write_receipt(out, args.json)
    print(f"receipt: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
