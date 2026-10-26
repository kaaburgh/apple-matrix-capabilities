#!/usr/bin/env python3
"""Disposable, source-controlled Phase 0b baseline experiment.

NOT an upstream receipt, not a GPU latency benchmark, and not a Metal
capture. Run uncaptured on real Apple Silicon after reviewing the protocol
in docs/lowbit-research/PHASE-0.md. Outputs are local-only experiments/.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


PROHIBITED_ENV = ("MTL_CAPTURE_ENABLED", "MTL_SHADER_VALIDATION",
                  "MTL_DEBUG_LAYER", "METAL_DEVICE_WRAPPER_TYPE")
NS = (1, 2, 4, 8, 16)
MIB = 1024 * 1024


def enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "off")


def validate_timing_env(env: dict[str, str]) -> None:
    problems = [name for name in PROHIBITED_ENV if enabled(env.get(name))]
    if problems:
        raise ValueError("GPU timing is invalid with enabled: " + ", ".join(problems))


def safe_environment(env: dict[str, str]) -> dict[str, str]:
    names = PROHIBITED_ENV + ("MLX_METAL_JIT", "MLX_ENABLE_TF32")
    allowed = {"unset", "0", "1", "off", "on", "true", "false"}
    return {name: (value if value in allowed else "set")
            for name in names
            for value in [(env.get(name, "unset") or "unset").strip().lower()]}


def eligible_ns(output_bytes: int, budget_bytes: int, candidates=NS):
    if output_bytes <= 0 or budget_bytes <= 0:
        raise ValueError("positive output size and memory budget required")
    choices = [n for n in candidates if n >= 1 and n * output_bytes <= budget_bytes]
    skipped = {str(n): "outputs_exceed_memory_budget" for n in candidates if n not in choices}
    return choices, skipped


def window_converged(means: list[float], tolerance: float = 0.02) -> bool:
    """Last-three-window per-dispatch times have <= tolerance range/median."""
    if len(means) < 3 or not 0 < tolerance < 1:
        return False
    last = means[-3:]
    if not all(math.isfinite(t) and t > 0 for t in last):
        return False
    return (max(last) - min(last)) / statistics.median(last) <= tolerance


def fit_tn(observations: list[tuple[int, float]]) -> dict:
    """OLS T(N)=a+bN, with residual-based uncertainty (not GPU-only latency)."""
    if len(observations) < 4 or len({n for n, _ in observations}) < 3:
        raise ValueError("need at least four timings across three distinct N")
    xs, ys = zip(*observations)
    if any(n < 1 for n in xs) or any(t <= 0 or not math.isfinite(t) for t in ys):
        raise ValueError("invalid batch observations")
    count = len(xs)
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in observations) / sxx
    intercept = ybar - slope * xbar
    residuals = [y - (intercept + slope * x) for x, y in observations]
    sigma2 = sum(v * v for v in residuals) / (count - 2)
    se_slope = math.sqrt(sigma2 / sxx)
    se_intercept = math.sqrt(sigma2 * (1 / count + xbar * xbar / sxx))
    sst = sum((y - ybar) ** 2 for y in ys)
    return {
        "a_effective_fixed_ms": intercept,
        "b_back_to_back_ms_per_dispatch": slope,
        "a_standard_error_ms": se_intercept,
        "b_standard_error_ms": se_slope,
        "residual_rms_ms": math.sqrt(sum(v * v for v in residuals) / count),
        "r_squared": 1 - sum(v * v for v in residuals) / sst if sst else None,
        "observations": count,
        "warning": "OLS uncertainty assumes independent samples; report order/correlation. "
                   "Not single-dispatch GPU latency.",
    }


def timed_batch(factory, synchronize, n: int) -> float:
    """Host-clock creation + one synchronization for N independent GPU outputs."""
    start = time.perf_counter_ns()
    outputs = [factory() for _ in range(n)]
    synchronize(*outputs)
    return (time.perf_counter_ns() - start) / 1e6


def warmup(factory, sync, *, n: int, min_s: float, max_s: float,
           window_s: float, tolerance: float) -> dict:
    if not (0 < window_s <= min_s <= max_s) or not 0 < tolerance < 1:
        raise ValueError("bad warmup configuration")
    start = time.perf_counter()
    means = []
    counts = []
    while True:
        wstart = time.perf_counter()
        calls, elapsed_ms = 0, 0.0
        while time.perf_counter() - wstart < window_s:
            elapsed_ms += timed_batch(factory, sync, n)
            calls += n
        means.append(elapsed_ms / calls)
        counts.append(calls)
        elapsed = time.perf_counter() - start
        if elapsed >= min_s and window_converged(means, tolerance):
            return {"converged": True, "elapsed_s": elapsed,
                    "window_ms_per_dispatch": means, "window_dispatches": counts,
                    "criterion": "last_3_range_over_median"}
        if elapsed >= max_s:
            return {"converged": False, "elapsed_s": elapsed,
                    "window_ms_per_dispatch": means, "window_dispatches": counts,
                    "criterion": "max_warmup_budget_exhausted"}


def data_summary(samples: list[float]) -> dict:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else None,
        "spread_max_over_min": max(samples) / min(samples),
    }


def measure_tn(factory, sync, ns: list[int], repetitions: int):
    if repetitions < 3 or not ns:
        raise ValueError("need >=3 repetitions and nonempty batch sizes")
    # Interleave batch sizes to reduce systematic power-state/order bias.
    by_n = {n: [] for n in ns}
    chronological = []
    for rep in range(repetitions):
        order = list(ns) if rep % 2 == 0 else list(reversed(ns))
        for n in order:
            elapsed = timed_batch(factory, sync, n)
            by_n[n].append(elapsed)
            chronological.append({"rep": rep, "n": n, "elapsed_ms": elapsed})
    return {
        "per_n": {str(n): data_summary(by_n[n]) for n in ns},
        "chronological": chronological,
        "fit": fit_tn([(r["n"], r["elapsed_ms"]) for r in chronological]),
        "timing_mode": "host_elapsed_construct_N_then_mx_eval_all",
    }


def interleaved_ab(a, b, sync, *, n: int, repetitions: int):
    if repetitions < 3 or n < 1:
        raise ValueError("AB: require >=3 repeats and positive n")
    samples = {"A": [], "B": []}
    chronological = []
    for rep in range(repetitions):
        for which in ("A", "B") if rep % 2 == 0 else ("B", "A"):
            ms = timed_batch(a if which == "A" else b, sync, n)
            samples[which].append(ms)
            chronological.append({"rep": rep, "tile": which, "elapsed_ms": ms})
    return {
        "n": n, "chronological": chronological,
        "A": data_summary(samples["A"]), "B": data_summary(samples["B"]),
        "median_ratio_B_over_A": (statistics.median(samples["B"]) /
                                   statistics.median(samples["A"])),
        "note": "Tile control only: identical lane/shape/precision, different tiles.",
    }


def checked_context(ls, *, lane: str, tile: tuple, shape: tuple, seed: int):
    import numpy as np
    rng = np.random.default_rng([seed, ls.LANE_INDEX[lane]])
    ok, entry, context = ls.check_tile(lane, tile, ls.CHK, rng, "lo", stage="check_shape")
    if not ok:
        raise ValueError("check_shape correctness failed")
    if shape != ls.CHK:
        del context
        ok, entry, context = ls.check_tile(lane, tile, shape, rng, "lo",
                                            stage="rate_shape")
        if not ok:
            raise ValueError("rate_shape correctness failed")
    def factory():
        return ls.call(context["kern"], context["inputs"], context["m"],
                       context["n"], tile)
    return factory, entry


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def run(args) -> dict:
    validate_timing_env(os.environ)
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        raise ValueError("Apple Silicon Mac required for real GPU timing")
    import mlx.core as mx
    import lane_sweep as ls
    if args.lane not in ls.LANES:
        raise ValueError("unknown upstream lane")
    if not 0 <= args.tile_a < len(ls.TILES):
        raise ValueError("invalid tile A index")
    if args.tile_b is not None and not 0 <= args.tile_b < len(ls.TILES):
        raise ValueError("invalid tile B index")
    shape = ls.CHK if args.shape == "check" else ls.RATE
    lane = args.lane
    tile_a = ls.TILES[args.tile_a]
    factory_a, check_a = checked_context(ls, lane=lane, tile=tile_a,
                                          shape=shape, seed=args.seed)

    device = mx.device_info()
    mem = int(device.get("memory_size", 0))
    output_bytes = shape[0] * shape[2] * 4
    # No single batch may allocate more than 512 MiB or ~10% of device RAM.
    limit = min(args.output_budget_mib * MIB, int(mem * 0.10)) if mem else args.output_budget_mib * MIB
    ns, skipped = eligible_ns(output_bytes, limit)
    if len(ns) < 3:
        raise ValueError("insufficient eligible N for T(N) fit at this shape/budget")
    warm_n = min(ns, key=lambda x: abs(x - args.warm_batch))
    warm_a = warmup(factory_a, mx.eval, n=warm_n, min_s=args.min_warmup_s,
                    max_s=args.max_warmup_s, window_s=args.window_s,
                    tolerance=args.tolerance)
    measurements = measure_tn(factory_a, mx.eval, ns, args.reps)

    root = Path(__file__).resolve().parents[1]
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("tools/phase0b_baseline.py", "probes/lane_sweep.py",
                            "probes/_common.py")}
    C = ls.C
    result = {
        "protocol": "phase0b-disposable-v1",
        "not_upstream_receipt": True,
        "observed": True,
        "device": {
            "name": device.get("device_name", "unknown"),
            "architecture": device.get("architecture", "unknown"),
            "memory_size": mem if mem else "unknown",
            "gpu_cores": C.gpu_cores(),
        },
        "macos": {"version": platform.mac_ver()[0],
                  "build": C.first_line(C.run_text(["sw_vers", "-buildVersion"]))},
        "toolchain": C.toolchain_meta(),
        "power": C.power_meta(),
        "mlx": package_version("mlx"), "numpy": package_version("numpy"),
        "environment": safe_environment(os.environ),
        "sources": sources,
        "lane": lane, "shape": list(shape),
        "tile_a": list(tile_a), "relaxed_precision": ls.lane_relaxed(lane),
        "check_a": C.finite_check(check_a),
        "warmup": warm_a,
        "steady_state_claim_allowed": warm_a["converged"],
        "output_bytes_per_dispatch": output_bytes,
        "batch_budget_bytes": limit,
        "skipped_batch_sizes": skipped,
        "batch_timing": measurements,
        "limitations": [
            "Host wall time includes MLX kernel-call construction and queue synchronization.",
            "Batch outputs may overlap; fitted b is not isolated GPU latency.",
            "No GPU clock, utilization, or physical ISA evidence is collected.",
            "Nonconverged results are diagnostic only, not steady-state benchmarks.",
        ],
    }
    if args.tile_b is not None:
        tile_b = ls.TILES[args.tile_b]
        factory_b, check_b = checked_context(ls, lane=lane, tile=tile_b,
                                              shape=shape, seed=args.seed)
        warm_b = warmup(factory_b, mx.eval, n=warm_n, min_s=args.min_warmup_s,
                        max_s=args.max_warmup_s, window_s=args.window_s,
                        tolerance=args.tolerance)
        result["tile_b"] = list(tile_b)
        result["check_b"] = C.finite_check(check_b)
        result["warmup_b"] = warm_b
        result["steady_state_claim_allowed"] = (warm_a["converged"] and warm_b["converged"])
        result["matched_tile_control"] = interleaved_ab(
            factory_a, factory_b, mx.eval, n=warm_n, repetitions=args.reps)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="local, new experiments/<chip>/...json file; not receipts/")
    p.add_argument("--shape", choices=("check", "rate"), default="rate")
    p.add_argument("--lane", default="f16*f16->f32")
    p.add_argument("--tile-a", type=int, default=0)
    p.add_argument("--tile-b", type=int, default=None,
                   help="optional A/B control tile index, using the same lane")
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--output-budget-mib", type=int, default=512)
    p.add_argument("--warm-batch", type=int, default=8)
    p.add_argument("--min-warmup-s", type=float, default=0.5)
    p.add_argument("--max-warmup-s", type=float, default=8.0)
    p.add_argument("--window-s", type=float, default=0.2)
    p.add_argument("--tolerance", type=float, default=0.02)
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        dest = args.output.resolve()
        if not dest.is_relative_to(root / "experiments") or dest.suffix != ".json":
            raise ValueError("output must be a JSON path below the gitignored experiments/ directory")
        if dest.exists() or dest.is_symlink():
            raise ValueError("output path exists; never overwrite experimental evidence")
        if not dest.parent.exists():
            raise ValueError("create the local experiments/<chip>/ directory first")
        if args.output_budget_mib <= 0 or args.reps < 3:
            raise ValueError("invalid budget/reps")
        sys.path.insert(0, str(root / "probes"))
        result = run(args)
        with dest.open("x") as fp:
            json.dump(result, fp, indent=2, allow_nan=False)
            fp.write("\n")
        print("Phase 0b finished; local report:", dest.name)
        print("Converged:", result["steady_state_claim_allowed"])
        return 0
    except Exception as exc:
        print("Phase 0b unsuccessful:", type(exc).__name__,
              "(see local diagnostics; no result was fabricated)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
