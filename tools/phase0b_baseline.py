#!/usr/bin/env python3
"""Source-controlled, uncaptured Phase 0b baseline protocol.

Separate GPU synchronization time from MLX lazy graph construction.
No upstream receipts, performance claims without real Apple Silicon, or
hardware counter/ISA claims. Local experiment outputs only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import traceback

PROHIBITED_ENV = ("MTL_CAPTURE_ENABLED", "MTL_SHADER_VALIDATION",
                  "MTL_DEBUG_LAYER", "METAL_DEVICE_WRAPPER_TYPE")
BUFFER_ENV = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")
NS = (1, 2, 4, 8, 16)
MIB = 1024 * 1024


class BlockedError(Exception):
    """Prerequisite not met; no measurement attempted."""


class PhaseFailure(Exception):
    """Exception with an explicit measurement phase, without publishing text."""

    def __init__(self, stage: str, cause: Exception):
        self.stage, self.cause = stage, cause
        super().__init__(str(cause))


@contextmanager
def phase(stage: str):
    try:
        yield
    except PhaseFailure:
        raise
    except Exception as exc:
        raise PhaseFailure(stage, exc) from exc


def enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "off")


def validate_timing_env(env: dict[str, str]) -> None:
    problems = [name for name in PROHIBITED_ENV if enabled(env.get(name))]
    if problems:
        raise BlockedError("instrumented GPU timing prohibited: " + ", ".join(problems))


def safe_environment(env: dict[str, str]) -> dict[str, str]:
    names = PROHIBITED_ENV + ("MLX_METAL_JIT", "MLX_ENABLE_TF32") + BUFFER_ENV
    allowed = {"unset", "0", "1", "off", "on", "true", "false"}
    result = {}
    for name in names:
        value = (env.get(name, "unset") or "unset").strip().lower()
        result[name] = value if value in allowed or value.isdigit() else "set"
    return result


def eligible_ns(output_bytes: int, budget_bytes: int, candidates=NS):
    if output_bytes <= 0 or budget_bytes <= 0:
        raise ValueError("positive output size and budget required")
    selected = [n for n in candidates if n >= 1 and n * output_bytes <= budget_bytes]
    omitted = {str(n): "outputs_exceed_memory_budget" for n in candidates
               if n not in selected}
    return selected, omitted


def window_converged(means: list[float], tolerance: float = 0.02) -> bool:
    if len(means) < 3 or not 0 < tolerance < 1:
        return False
    last = means[-3:]
    if not all(math.isfinite(t) and t > 0 for t in last):
        return False
    return (max(last) - min(last)) / statistics.median(last) <= tolerance


def timed_batch(factory, synchronize, n: int) -> dict:
    """Return separate graph construction and eval+sync host-clock intervals.

    Graph construction might itself trigger MLX scheduler behavior; eval time
    is still *host observed*, not a GPU-side duration.
    """
    if n < 1:
        raise ValueError("n must be positive")
    t0 = time.perf_counter_ns()
    outputs = [factory() for _ in range(n)]
    t1 = time.perf_counter_ns()
    synchronize(*outputs)
    t2 = time.perf_counter_ns()
    return {"eval_ms": (t2 - t1) / 1e6, "construct_ms": (t1 - t0) / 1e6}


def warmup(factory, sync, *, n: int, min_s: float, max_s: float,
           window_s: float, tolerance: float) -> dict:
    if not (0 < window_s <= min_s <= max_s) or not 0 < tolerance < 1:
        raise ValueError("bad warmup configuration")
    start = time.perf_counter()
    means, counts, batches = [], [], []
    while True:
        wstart = time.perf_counter()
        calls, total_eval_ms = 0, 0.0
        while time.perf_counter() - wstart < window_s:
            offset = time.perf_counter() - start
            sample = timed_batch(factory, sync, n)
            batches.append({"offset_s": offset, "n": n, **sample})
            total_eval_ms += sample["eval_ms"]
            calls += n
        means.append(total_eval_ms / calls)
        counts.append(calls)
        elapsed = time.perf_counter() - start
        if elapsed >= min_s and window_converged(means, tolerance):
            return {"converged": True, "elapsed_s": elapsed,
                    "window_eval_ms_per_dispatch": means,
                    "window_dispatches": counts, "raw_batches": batches,
                    "criterion": "last_3_range_over_median"}
        if elapsed >= max_s:
            return {"converged": False, "elapsed_s": elapsed,
                    "window_eval_ms_per_dispatch": means,
                    "window_dispatches": counts, "raw_batches": batches,
                    "criterion": "max_warmup_budget_exhausted"}


def fit_tn(observations: list[tuple[int, float]]) -> dict:
    """OLS fit; residual SE is *secondary* and assumes independent samples."""
    if len(observations) < 3 or len({n for n, _ in observations}) < 3:
        raise ValueError("need timings at >=3 distinct N")
    xs, ys = zip(*observations)
    if any(n < 1 for n in xs) or any(t < 0 or not math.isfinite(t) for t in ys):
        raise ValueError("invalid batch observations")
    count = len(xs)
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in observations) / sxx
    intercept = ybar - slope * xbar
    residuals = [y - (intercept + slope * x) for x, y in observations]
    rss = sum(v * v for v in residuals)
    sigma2 = rss / (count - 2)
    sst = sum((y - ybar) ** 2 for y in ys)
    return {
        "a_effective_fixed_ms": intercept,
        "b_back_to_back_ms_per_dispatch": slope,
        "a_ols_standard_error_ms": math.sqrt(sigma2 * (1/count + xbar*xbar/sxx)),
        "b_ols_standard_error_ms": math.sqrt(sigma2 / sxx),
        "r_squared": 1 - rss / sst if sst else None,
        "observations": count,
        "warning": "Secondary pooled OLS assumes independent homoscedastic samples; "
                   "use per-repetition slope distribution for uncertainty. "
                   "Not single-dispatch GPU latency.",
    }


def quantile_sorted(sorted_values: list[float], proportion: float) -> float:
    """Interpolated percentile, explicitly descriptive (not confidence interval)."""
    if not sorted_values or not 0 <= proportion <= 1:
        raise ValueError("invalid quantile")
    p = proportion * (len(sorted_values) - 1)
    lo, hi = math.floor(p), math.ceil(p)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (p - lo)


def data_summary(samples: list[float]) -> dict:
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("nonfinite/nonpositive timing sample")
    return {
        "samples_ms": samples, "median_ms": statistics.median(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else None,
        "spread_max_over_min": max(samples)/min(samples),
    }


def measure_tn(factory, sync, ns: list[int], repetitions: int):
    if repetitions < 3 or len(set(ns)) < 3:
        raise ValueError("need >=3 repetitions and >=3 eligible N")
    samples = {n: {"eval_ms": [], "construct_ms": []} for n in ns}
    chronological, per_repetition = [], []
    origin = time.perf_counter()
    for rep in range(repetitions):
        order = list(ns) if rep % 2 == 0 else list(reversed(ns))
        row = []
        for n in order:
            offset = time.perf_counter() - origin
            obs = timed_batch(factory, sync, n)
            for key in ("eval_ms", "construct_ms"):
                samples[n][key].append(obs[key])
            chronological.append({"rep": rep, "n": n, "offset_s": offset, **obs})
            row.append((n, obs["eval_ms"]))
        fit = fit_tn(row)
        per_repetition.append({"rep": rep,
                               "order": "asc" if rep % 2 == 0 else "desc",
                               "a_ms": fit["a_effective_fixed_ms"],
                               "b_ms_per_dispatch": fit["b_back_to_back_ms_per_dispatch"]})
    slopes = sorted(x["b_ms_per_dispatch"] for x in per_repetition)
    by_order = {
        direction: [f["b_ms_per_dispatch"] for f in per_repetition
                    if f["order"] == direction]
        for direction in ("asc", "desc")
    }
    medians_by_order = {
        direction: statistics.median(values)
        for direction, values in by_order.items() if values
    }
    all_samples = [(r["n"], r["eval_ms"]) for r in chronological]
    return {
        "per_n": {str(n): {k: data_summary(v) for k, v in sample.items()}
                  for n, sample in samples.items()},
        "chronological": chronological,
        "fit_pooled_secondary": fit_tn(all_samples),
        "per_repetition_fits": per_repetition,
        "b_repetition_distribution": {
            "median_ms_per_dispatch": statistics.median(slopes),
            "median_by_order_ms_per_dispatch": medians_by_order,
            "repetitions_by_order": {direction: len(values)
                                     for direction, values in by_order.items()},
            "order_gap_desc_minus_asc_ms_per_dispatch": (
                medians_by_order["desc"] - medians_by_order["asc"]
                if len(medians_by_order) == 2 else None),
            "min_ms_per_dispatch": min(slopes),
            "max_ms_per_dispatch": max(slopes),
            "p025_p975_descriptive_ms": [quantile_sorted(slopes, .025),
                                         quantile_sorted(slopes, .975)],
            "warning": "Percentiles are descriptive across repeats, NOT a "
                       "coverage-calibrated confidence interval. Asc/desc "
                       "differences may reveal drift rather than independent noise.",
        },
        "timing_mode": "construct_lazy_graph_outside_timer_then_time_mx_eval_and_sync",
    }


def interleaved_ab(a, b, sync, *, n: int, repetitions: int):
    if repetitions < 3 or n < 1:
        raise ValueError("AB requires >=3 repeats and positive n")
    samples = {"A": {"eval_ms": [], "construct_ms": []},
               "B": {"eval_ms": [], "construct_ms": []}}
    chronological = []
    origin = time.perf_counter()
    for rep in range(repetitions):
        for which in (("A", "B") if rep % 2 == 0 else ("B", "A")):
            offset = time.perf_counter() - origin
            obs = timed_batch(a if which == "A" else b, sync, n)
            for key in ("eval_ms", "construct_ms"):
                samples[which][key].append(obs[key])
            chronological.append({"rep": rep, "tile": which, "offset_s": offset, **obs})
    return {
        "n": n, "chronological": chronological,
        "A": {k: data_summary(v) for k, v in samples["A"].items()},
        "B": {k: data_summary(v) for k, v in samples["B"].items()},
        "median_ratio_B_over_A_eval": (statistics.median(samples["B"]["eval_ms"]) /
                                        statistics.median(samples["A"]["eval_ms"])),
        "note": "Matched lane/shape/precision; tile geometry is the only intended difference.",
    }


def validate_tile_shape(shape, tile):
    m, _, n = shape
    mt, nt, sg = tile
    if mt <= 0 or nt <= 0 or sg not in (1, 2, 4, 8) or m % mt or n % nt:
        raise ValueError("selected tile does not divide chosen output geometry")


def checked_context(ls, *, lane: str, tile: tuple, shape: tuple, seed: int):
    """Upstream CHK always uses canonical TILES[0], then RATE target tile.

    For --shape check, reject unsupported tiles rather than dispatching zero
    threadgroups. Returns context for optional original-protocol replica.
    """
    import numpy as np
    validate_tile_shape(shape, tile)
    rng = np.random.default_rng([seed, ls.LANE_INDEX[lane]])
    ok, entry, context = ls.check_tile(lane, ls.TILES[0], ls.CHK, rng, "lo",
                                       stage="check_shape")
    if not ok:
        raise PhaseFailure("check_shape", ValueError("upstream check_shape failed"))
    if shape != ls.CHK or tile != ls.TILES[0]:
        # Even the quick CHK diagnostic must test the actual selected tile,
        # not time a kernel constructed for canonical TILES[0].
        second_stage = "rate_shape" if shape != ls.CHK else "check_shape_tile"
        del context
        try:
            ok, entry, context = ls.check_tile(lane, tile, shape, rng, "lo",
                                                stage=second_stage)
        except Exception as exc:
            raise PhaseFailure(second_stage, exc) from exc
        if not ok:
            raise PhaseFailure(second_stage, ValueError("target tile correctness failed"))

    def factory():
        return ls.call(context["kern"], context["inputs"], context["m"],
                       context["n"], tile)
    return factory, entry, context


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def run(args) -> dict:
    with phase("preflight"):
        validate_timing_env(os.environ)
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            raise BlockedError("Apple Silicon Mac required")
        if args.reps < 3:
            raise ValueError("minimum 3 repetitions required")
        import mlx.core as mx
        import lane_sweep as ls
        if args.lane not in ls.LANES:
            raise ValueError("unknown upstream lane")
        if not 0 <= args.tile_a < len(ls.TILES):
            raise ValueError("invalid tile A index")
        if args.tile_b is not None and not 0 <= args.tile_b < len(ls.TILES):
            raise ValueError("invalid tile B index")
        shape = ls.CHK if args.shape == "check" else ls.RATE
        for index in (args.tile_a, args.tile_b):
            if index is not None:
                validate_tile_shape(shape, ls.TILES[index])
        lane = args.lane
        tile_a = ls.TILES[args.tile_a]
    with phase("check_shape"):
        factory_a, check_a, ctx_a = checked_context(
            ls, lane=lane, tile=tile_a, shape=shape, seed=args.seed)

    with phase("budget"):
        device = mx.device_info()
        mem = int(device.get("memory_size", 0))
        output_bytes = shape[0] * shape[2] * 4
        limit = min(args.output_budget_mib * MIB, int(mem * .10)) if mem else args.output_budget_mib * MIB
        ns, skipped = eligible_ns(output_bytes, limit)
        if len(ns) < 3:
            raise BlockedError("budget allows fewer than 3 batch sizes; increase budget or use check shape")
        warm_n = min(ns, key=lambda x: abs(x - args.warm_batch))

    # This optional replica MUST precede sustained warmup, so the first
    # time_tile() sample captures original post-check GPU behavior.
    upstream = None
    if args.upstream_replica:
        with phase("upstream_replica"):
            upstream = {
                "warmups": 2, "reps": 5, "tile": list(tile_a),
                "samples_ms": ls.time_tile(ctx_a, tile_a, reps=5, warmups=2),
                "warning": "Includes per-call graph construction, as in stock lane_sweep.",
            }

    with phase("warmup"):
        warm_a = warmup(factory_a, mx.eval, n=warm_n, min_s=args.min_warmup_s,
                        max_s=args.max_warmup_s, window_s=args.window_s,
                        tolerance=args.tolerance)
    with phase("measure"):
        measurements = measure_tn(factory_a, mx.eval, ns, args.reps)

    root = Path(__file__).resolve().parents[1]
    C = ls.C
    with phase("metadata"):
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("tools/phase0b_baseline.py", "probes/lane_sweep.py",
                                "probes/_common.py")}
        result = {
            "status": "observed", "protocol": "phase0b-disposable-v2",
            "not_upstream_receipt": True,
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
            "upstream_protocol_replica": upstream,
            "warmup": warm_a, "steady_state_claim_allowed": warm_a["converged"],
            "output_bytes_per_dispatch": output_bytes,
            "batch_budget_bytes": limit, "skipped_batch_sizes": skipped,
            "batch_timing": measurements,
            "limitations": [
                "Host-clock MLX graph construction and evaluation/synchronization are separate.",
                "Synchronized eval is not a direct GPU device-side timing.",
                "Batch outputs can overlap; fitted b is not GPU dispatch latency.",
                "Pooled OLS standard errors assume independence and are secondary.",
                "No GPU clock, utilization or physical ISA evidence is collected.",
                "Nonconverged results are diagnostic only, not steady-state benchmarks.",
            ],
        }

    if args.tile_b is not None:
        with phase("check_shape"):
            tile_b = ls.TILES[args.tile_b]
            factory_b, check_b, _ = checked_context(
                ls, lane=lane, tile=tile_b, shape=shape, seed=args.seed)
        with phase("warmup_b"):
            warm_b = warmup(factory_b, mx.eval, n=warm_n, min_s=args.min_warmup_s,
                            max_s=args.max_warmup_s, window_s=args.window_s,
                            tolerance=args.tolerance)
        with phase("measure_ab"):
            ab = interleaved_ab(factory_a, factory_b, mx.eval, n=warm_n,
                                repetitions=args.reps)
        result.update({
            "tile_b": list(tile_b), "check_b": C.finite_check(check_b),
            "warmup_b": warm_b,
            "steady_state_claim_allowed": warm_a["converged"] and warm_b["converged"],
            "matched_tile_control": ab,
        })
    return result


def classify_error(exc: Exception) -> str:
    exc = exc.cause if isinstance(exc, PhaseFailure) else exc
    try:
        import _common as C
        return C.classify(exc)
    except Exception:
        return {"ValueError": "value", "RuntimeError": "runtime",
                "PermissionError": "permission"}.get(type(exc).__name__, "other")


def store_exclusive_json(dest: Path, record: dict) -> None:
    """Serialize before writing; temp+hardlink publishes atomically, no overwrite."""
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=".phase0b-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, dest)  # no replacement if destination already exists
    finally:
        os.unlink(temp)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="new local experiments/<chip>/...json; never receipts/")
    p.add_argument("--shape", choices=("check", "rate"), default="rate")
    p.add_argument("--lane", default="f16*f16->f32")
    p.add_argument("--tile-a", type=int, default=0)
    p.add_argument("--tile-b", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--output-budget-mib", type=int, default=512)
    p.add_argument("--warm-batch", type=int, default=8)
    p.add_argument("--min-warmup-s", type=float, default=0.5)
    p.add_argument("--max-warmup-s", type=float, default=8.0)
    p.add_argument("--window-s", type=float, default=0.2)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--upstream-replica", action="store_true",
                   help="run original 2-warmup/5-rep protocol immediately after correctness")
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    dest = args.output.resolve()
    stage = "preflight"
    try:
        with phase("preflight"):
            if not dest.is_relative_to(root / "experiments") or dest.suffix != ".json":
                raise BlockedError("output must be a JSON path under gitignored experiments/")
            if dest.exists() or dest.is_symlink():
                raise BlockedError("refusing to overwrite an existing experiment")
            if not dest.parent.is_dir():
                raise BlockedError("create experiments/<chip>/ directory first")
            if args.output_budget_mib <= 0 or args.reps < 3:
                raise ValueError("invalid output budget/repetitions")
            sys.path.insert(0, str(root / "probes"))
        result = run(args)
        with phase("write"):
            store_exclusive_json(dest, result)
        print("Phase 0b local result:", dest.name)
        print("Steady state:", result["steady_state_claim_allowed"])
        return 0
    except Exception as exc:
        underlying = exc.cause if isinstance(exc, PhaseFailure) else exc
        stage = exc.stage if isinstance(exc, PhaseFailure) else stage
        blocked = isinstance(underlying, BlockedError)
        failure = {
            "status": "blocked" if blocked else "failed",
            "stage": stage,
            "not_upstream_receipt": True,
            "error": {"type": type(underlying).__name__,
                      "category": classify_error(exc)},
        }
        # The local console may contain detailed diagnostics, but never copy
        # traceback or private filesystem names into the report.
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        print(json.dumps(failure, sort_keys=True))
        if dest.is_relative_to(root / "experiments") and dest.parent.is_dir() and not dest.exists():
            try:
                store_exclusive_json(dest, failure)
            except Exception:
                # Error JSON is printed regardless of destination failure.
                pass
        return 2 if blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
