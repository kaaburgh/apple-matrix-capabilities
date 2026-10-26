#!/usr/bin/env python3
"""PR A reusable float-lane capture and live GPU loop. Local evidence only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import phase0a_capture_spike as A
import phase0b_baseline as B

SUPPORTED = (
    "f16*f16->f32", "e4m3*e4m3->f32", "fp4*fp4->f32",
    "f32*f32->f32", "strict_f32*f32->f32",
)
SOURCES = (
    "probes/capture_lane.py", "probes/lane_sweep.py", "probes/_common.py",
    "tools/phase0a_capture_spike.py", "tools/phase0b_baseline.py",
)


def validate_paths(output, trace):
    """Only new, local gitignored experimental evidence may be written."""
    base = (ROOT / "experiments").resolve()
    dest = output.resolve()
    if dest.suffix != ".json" or not dest.is_relative_to(base):
        raise B.BlockedError("manifest must be .json under experiments/")
    if not dest.parent.is_dir() or dest.exists() or output.is_symlink():
        raise B.BlockedError("output path exists or parent missing")
    if trace is None:
        return dest, None
    target = trace.resolve()
    if not target.is_relative_to(base) or target.suffix != ".gputrace":
        raise B.BlockedError("trace must be .gputrace under experiments/")
    if not target.parent.is_dir() or target.exists() or trace.is_symlink():
        raise B.BlockedError("trace path exists or parent missing")
    return dest, target


def validate_mode(args, env, ls, trace):
    if args.lane not in SUPPORTED or args.lane not in ls.LANES:
        raise B.BlockedError("unsupported lane for initial PR A")
    if not 0 <= args.tile < len(ls.TILES):
        raise B.BlockedError("invalid upstream tile index")
    shape = ls.CHK if args.shape == "check" else ls.RATE
    tile = ls.TILES[args.tile]
    B.validate_tile_shape(shape, tile)
    if not 1 <= args.batch <= 16 or args.output_budget_mib < 1:
        raise B.BlockedError("batch must be 1..16 and output budget positive")
    if trace is not None:
        A.validate_preflight(platform.system(), platform.machine(),
                             env.get("MTL_CAPTURE_ENABLED"), trace)
    else:
        if args.loop is None or not 0 < args.loop <= 600:
            raise B.BlockedError("loop duration must be 0 < seconds <= 600")
        B.validate_timing_env(env)
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            raise B.BlockedError("Apple Silicon Mac required")
    return shape, tile


def allocation_guard(shape, memory, batch, mib):
    outbytes = shape[0] * shape[2] * 4
    limit = min(mib * B.MIB, int(memory * .10)) if memory else mib * B.MIB
    options, _ = B.eligible_ns(outbytes, limit, [batch])
    if not options:
        raise B.BlockedError("batch output arrays exceed memory budget")
    return {
        "bytes_per_output": outbytes, "batch_output_bytes": outbytes * batch,
        "budget_bytes": limit, "batch_selected": batch,
        "upstream_allocator": "lane_sweep.cap_memory: memory <=12GiB, cache<=2GiB",
    }


def checked_context(ls, np, lane, tile, shape):
    """Upstream intended-value correctness, lo then optional hi packing retry."""
    failed = []
    for order in ("lo", "hi"):
        rng = np.random.default_rng([ls.SEED, ls.LANE_INDEX[lane]])
        checks = []
        ok, entry, ctx = ls.check_tile(
            lane, ls.TILES[0], ls.CHK, rng, order, stage="check_shape")
        checks.append(ls.C.finite_check(entry))
        if not ok:
            failed.append({"order": order, "stage": "check_shape"})
            continue
        if shape != ls.CHK or tile != ls.TILES[0]:
            del ctx
            stage = "rate_shape" if shape != ls.CHK else "check_shape_tile"
            ok, entry, ctx = ls.check_tile(lane, tile, shape, rng, order, stage=stage)
            checks.append(ls.C.finite_check(entry))
            if not ok:
                failed.append({"order": order, "stage": stage})
                continue
        return ctx, {"packing": order, "checks": checks, "failed_orders": failed}
    raise ValueError("upstream intended-value checks failed for both packing orders")


def capture_batch(mx, factory, trace, batch, mode):
    """Reuse verified Phase 0a 1-dispatch capture, or evaluate N in same window."""
    if batch == 1:
        return [A.capture_once(mx, factory, trace, mode=mode)]
    if mode not in ("fresh-stream", "default-stream"):
        raise ValueError("unsupported capture mode")
    mx.metal.start_capture(str(trace))
    try:
        if mode == "fresh-stream":
            stream = mx.new_stream(mx.gpu)
            with mx.stream(stream):
                outputs = [factory() for _ in range(batch)]
                mx.eval(*outputs)
        else:
            mx.synchronize(mx.default_stream(mx.gpu))
            outputs = [factory() for _ in range(batch)]
            mx.eval(*outputs)
        return outputs
    finally:
        mx.metal.stop_capture()


def loop_batches(factory, synchronize, seconds, batch, warmup_config):
    """Timed, uncaptured batches bracketed by idle gaps for Instruments."""
    warm = B.warmup(factory, synchronize, n=batch, **warmup_config)
    time.sleep(.25)
    start = time.perf_counter_ns()
    samples = []
    while (time.perf_counter_ns() - start) / 1e9 < seconds:
        offset = (time.perf_counter_ns() - start) / 1e9
        samples.append({"offset_s": offset, "n": batch,
                        **B.timed_batch(factory, synchronize, batch)})
    elapsed = (time.perf_counter_ns() - start) / 1e9
    time.sleep(.25)
    eval_ms = sum(r["eval_ms"] for r in samples)
    return {
        "warmup": warm, "converged": bool(warm["converged"]),
        "steady_state_claim_allowed": bool(warm["converged"]),
        "loop": {
            "seconds_requested": seconds, "seconds_host_elapsed": elapsed,
            "batch": batch, "batches": len(samples),
            "dispatches": batch * len(samples), "samples": samples,
            "total_eval_ms": eval_ms,
            "total_construct_ms": sum(r["construct_ms"] for r in samples),
            "pre_post_idle_gap_s": .25,
            "gpu_idle_fraction": None,
            "gpu_idle_fraction_source": "not_measured_instruments_required",
            "neural_accelerator_utilization": None,
            "host_eval_dispatches_per_s": (
                1000 * batch * len(samples) / eval_ms if eval_ms > 0 else None),
        },
    }


def run(args, env=None):
    if env is None:
        env = os.environ
    stage = "preflight"
    try:
        dest, trace = validate_paths(args.output, args.capture)
        if trace is not None:
            A.validate_preflight(platform.system(), platform.machine(),
                                 env.get("MTL_CAPTURE_ENABLED"), trace)
        else:
            B.validate_timing_env(env)
            if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
                raise B.BlockedError("Apple Silicon Mac required")
        sys.path.insert(0, str(ROOT / "probes"))
        import numpy as np
        import mlx.core as mx
        import lane_sweep as ls  # import deliberately invokes cap_memory
        shape, tile = validate_mode(args, env, ls, trace)
        stage = "memory_budget"
        device = mx.device_info()
        memory = int(device.get("memory_size") or 0)
        guard = allocation_guard(shape, memory, args.batch, args.output_budget_mib)
        stage = "toolchain"
        xcode = A.xcode_version() if trace else "not_required"
        cli = A.metal_toolchain() if trace else "not_checked"
        stage = "correctness"
        ctx, checks = checked_context(ls, np, args.lane, tile, shape)
        factory = lambda: ls.call(ctx["kern"], ctx["inputs"], ctx["m"], ctx["n"], tile)
        tag = "".join(c if c.isalnum() else "_" for c in args.lane)
        expected = f"amc_{tag}_{shape[0]}x{shape[1]}x{shape[2]}_{tile[0]}_{tile[1]}_{tile[2]}"
        stage = "provenance"
        manifest = A._machine_metadata(mx, ls, ROOT)
        manifest["sources"] = {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
        manifest.update({
            "protocol": "pr-a-float-capture-loop-v1", "not_upstream_receipt": True,
            "mode": "capture" if trace else "live_loop",
            "lane": args.lane, "shape": list(shape), "tile": list(tile),
            "packing": checks["packing"], "checks": checks["checks"],
            "failed_packing_attempts": checks["failed_orders"],
            "lane_seed_index": ls.LANE_INDEX[args.lane],
            "lane_definition": list(ls.LANES[args.lane]),
            "relaxed_precision": bool(ls.lane_relaxed(args.lane)),
            "expected_kernel_name_contains": expected,
            "dispatches_requested": args.batch,
            "capture_mode": args.capture_mode if trace else None,
            "xcode_version": xcode, "metal_cli": cli,
            "environment": B.safe_environment(env), "memory": guard,
            "replay": {"verified": False, "settings": None,
                       "command_buffers": None, "dispatch_calls": None} if trace else None,
            "gpu_counter_scope": "not_measured",
        })
        if trace is not None:
            stage = "jit_warmup"
            precapture = np.array(factory())
            mx.eval(factory())
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            stage = "capture"
            results = capture_batch(mx, factory, trace, args.batch, args.capture_mode)
            if not trace.exists():
                raise RuntimeError("capture bundle not created")
            stage = "post_capture_check"
            if not all(np.array_equal(np.array(x), precapture) for x in results):
                raise RuntimeError("captured output mismatch")
            manifest.update({
                "status": "capture_written_replay_not_verified",
                "captured_equals_precapture": True, "trace_basename": trace.name,
                "command_buffers_verified": False,
                "convergence_not_applicable_to_replay": True,
            })
        else:
            stage = "warmup_and_loop"
            settings = dict(min_s=args.min_warmup_s, max_s=args.max_warmup_s,
                            window_s=args.window_s, tolerance=args.tolerance)
            manifest.update(loop_batches(factory, mx.eval, args.loop, args.batch, settings))
            manifest["status"] = ("observed_converged" if manifest["converged"]
                                  else "observed_nonconverged_diagnostic")
        return manifest
    except Exception as exc:
        return {
            "status": ("blocked" if isinstance(
                exc, (B.BlockedError, A.PreflightError, ImportError)) else "failed"),
            "stage": stage, "error": {"type": type(exc).__name__,
                                     "category": A._category(exc)},
            "not_upstream_receipt": True,
            "mode": "capture" if args.capture is not None else "live_loop",
        }


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", type=Path, metavar="TRACE")
    mode.add_argument("--loop", type=float, metavar="SECONDS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=SUPPORTED, required=True)
    parser.add_argument("--shape", choices=("check", "rate"), default="rate")
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--capture-mode", choices=("fresh-stream", "default-stream"),
                        default="fresh-stream")
    parser.add_argument("--output-budget-mib", type=int, default=512)
    parser.add_argument("--min-warmup-s", type=float, default=.5)
    parser.add_argument("--max-warmup-s", type=float, default=8.)
    parser.add_argument("--window-s", type=float, default=.2)
    parser.add_argument("--tolerance", type=float, default=.02)
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    if args.batch is None:
        args.batch = 1 if args.capture is not None else 8
    report = run(args)
    try:
        # A successful capture legitimately created the trace already. On
        # publication recheck only the JSON destination, never the trace.
        dest, _ = validate_paths(args.output, None)
        B.store_exclusive_json(dest, report)
    except Exception:
        print("Local manifest could not be written; no evidence published", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "mode": report["mode"],
                      "manifest_name": dest.name}, sort_keys=True))
    return (2 if report["status"] == "blocked"
            else 1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
