#!/usr/bin/env python3
"""Phase 0a MLX Metal capture experiment, not a benchmark.

On a target device, --shape rate profiles the upstream standard geometry.
Hosted M1 CI uses --shape check to exercise only capture plumbing.
No captured trace proves Neural Accelerator use until inspected in Xcode.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


class PreflightError(RuntimeError):
    """Missing hardware, tools or opt-in; reports blocked rather than failed."""


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "off")


def environment() -> dict[str, str]:
    names = ("MTL_CAPTURE_ENABLED", "MTL_SHADER_VALIDATION", "MTL_DEBUG_LAYER",
             "METAL_DEVICE_WRAPPER_TYPE", "MLX_METAL_JIT", "MLX_ENABLE_TF32")
    return {name: os.environ.get(name, "unset") for name in names}


def validate_preflight(system: str, architecture: str,
                       capture_enabled: str | None, trace_path: Path) -> None:
    if system != "Darwin" or architecture not in ("arm64", "aarch64"):
        raise PreflightError("Apple Silicon Mac required")
    if capture_enabled != "1":
        raise PreflightError("MTL_CAPTURE_ENABLED=1 required at process start")
    if _enabled(os.environ.get("MTL_SHADER_VALIDATION")):
        raise PreflightError("MTL_SHADER_VALIDATION must be disabled for uninstrumented capture")
    if trace_path.exists() or trace_path.is_symlink():
        raise PreflightError("Trace path already exists; choose a new .gputrace path")
    if trace_path.suffix != ".gputrace":
        raise PreflightError("Trace output must end in .gputrace")
    if not trace_path.parent.is_dir():
        raise PreflightError("Trace parent directory must already exist")


def xcode_version() -> str:
    try:
        result = subprocess.run(["xcodebuild", "-version"], text=True,
                                capture_output=True, check=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise PreflightError("Full Xcode 27 unavailable") from exc
    lines = result.stdout.splitlines()
    first = lines[0] if lines else ""
    match = re.match(r"Xcode\s+(\d+)(?:\.|$)", first)
    if not match or int(match.group(1)) < 27:
        raise PreflightError("Full Xcode 27 or later required")
    return result.stdout.strip()


def metal_toolchain() -> None:
    try:
        subprocess.run(["xcrun", "--sdk", "macosx", "--find", "metal"],
                       text=True, capture_output=True, check=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise PreflightError("Metal Toolchain component unavailable") from exc


def source_hashes(repo_root: Path) -> dict[str, str]:
    paths = ("tools/phase0a_capture_spike.py", "probes/lane_sweep.py",
             "probes/_common.py")
    return {p: hashlib.sha256((repo_root / p).read_bytes()).hexdigest()
            for p in paths}


def capture_once(mx, dispatch, trace_path: Path):
    """Capture exactly one evaluated invocation; return its result for checking."""
    mx.metal.start_capture(str(trace_path))
    try:
        result = dispatch()
        mx.eval(result)
        return result
    finally:
        mx.metal.stop_capture()


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _category(exc: Exception) -> str:
    # Same controlled categories used by Precisit's receipt error objects.
    root = Path(__file__).resolve().parent.parent
    p = str(root / "probes")
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        import _common as C
        return C.classify(exc)
    except Exception:
        return "other"


def _machine_metadata(mx, ls, root: Path) -> dict:
    # Original helpers return "unknown" when specific hardware data is absent.
    C = ls.C
    info = mx.device_info()
    toolchain = C.toolchain_meta()
    macos = platform.mac_ver()
    return {
        "mlx": _version("mlx"),
        "os": {"version": macos[0], "build": C.first_line(C.run_text(["sw_vers", "-buildVersion"]))},
        "toolchain": toolchain,
        "device": {
            "name": info.get("device_name", "unknown"),
            "architecture": info.get("architecture", "unknown"),
            "memory_size": info.get("memory_size", "unknown"),
            "gpu_cores": C.gpu_cores(),
        },
        "power": C.power_meta(),
        "sources": source_hashes(root),
    }


def run(trace_path: Path, shape: str = "rate") -> dict:
    """Return a sanitized JSON-ready report on every normal/exceptional exit.

    BLOCKED: environment/SDK unsuitable. FAILED: experiment attempted and did
    not pass. TRACE_WRITTEN: no replay or hardware counters have been verified.
    """
    if shape not in ("check", "rate"):
        raise ValueError("shape must be check or rate")
    stage = "preflight"
    report = {
        "status": "blocked", "stage": stage, "shape_choice": shape,
        "trace_basename": trace_path.name, "env": environment(),
        "trace_replayed_in_xcode": False, "counters_collected": False,
    }
    try:
        validate_preflight(platform.system(), platform.machine(),
                           os.environ.get("MTL_CAPTURE_ENABLED"), trace_path)
        stage = "toolchain"
        xc = xcode_version()
        metal_toolchain()
        report["xcode"] = xc.splitlines()[0]
        report["xcode_build"] = xc.splitlines()[1] if len(xc.splitlines()) > 1 else "unknown"

        stage = "dependencies"
        root = Path(__file__).resolve().parent.parent
        if str(root / "probes") not in sys.path:
            sys.path.insert(0, str(root / "probes"))
        try:
            import mlx.core as mx
            import numpy as np
            import lane_sweep as ls
        except ImportError as exc:
            raise PreflightError("Required mlx/numpy/probe imports unavailable") from exc

        if not hasattr(mx, "metal") or not all(
            callable(getattr(mx.metal, name, None))
            for name in ("start_capture", "stop_capture")
        ):
            raise PreflightError("MLX Metal capture APIs unavailable")

        # Gather complete provenance before capture; absent optional telemetry
        # is unknown, and cannot invalidate a successfully created trace.
        report.update(_machine_metadata(mx, ls, root))
        report["numpy"] = np.__version__
        lane = "f16*f16->f32"
        tile = ls.TILES[0]  # 64x128x4
        rng = np.random.default_rng([ls.SEED, ls.LANE_INDEX[lane]])
        shape_tuple = ls.CHK if shape == "check" else ls.RATE
        report.update(lane=lane, shape=list(shape_tuple), tile=list(tile),
                      dispatches_requested=1)

        stage = "check"
        if shape == "rate":
            check_ok, check_entry, check_ctx = ls.check_tile(
                lane, tile, ls.CHK, rng, "lo", stage="check_shape")
            report["precheck"] = ls.C.finite_check(check_entry)
            if not check_ok:
                raise RuntimeError("Upstream check-shape correctness failed")
            del check_ctx

        good, entry, ctx = ls.check_tile(lane, tile, shape_tuple, rng, "lo",
                                         stage="rate_shape" if shape == "rate" else "check_shape")
        report["check"] = ls.C.finite_check(entry)
        if not good:
            raise RuntimeError("Upstream target-shape correctness failed")

        def dispatch():
            return ls.call(ctx["kern"], ctx["inputs"], ctx["m"], ctx["n"], tile)

        # Validate one of the same dispatches before capture; compare the
        # captured result after capture with this verified invocation.
        validated = np.array(dispatch())
        mx.eval(dispatch())   # short compile/first-touch warm-up (not steady-state)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()  # cached intermediates only; ctx retains GPU inputs

        stage = "capture"
        captured = capture_once(mx, dispatch, trace_path)
        if not trace_path.exists():
            raise RuntimeError("Capture finished without creating .gputrace")

        stage = "captured_output_check"
        actual = np.array(captured)
        if not np.array_equal(validated, actual):
            raise RuntimeError("Captured dispatch differs from verified dispatch")

        report["status"] = "capture_written_replay_not_verified"
        report["stage"] = "done"
        report["captured_output_equal"] = True
        return report
    except Exception as exc:
        report["status"] = "blocked" if isinstance(exc, PreflightError) else "failed"
        report["stage"] = stage
        report["error"] = {"type": type(exc).__name__, "category": _category(exc)}
        # Deliberately omit arbitrary exception text: compiler errors can
        # contain private file paths, hostnames and environment details.
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("phase0a-f16.gputrace"),
                        help="new .gputrace bundle path (must not exist)")
    parser.add_argument("--shape", choices=("check", "rate"), default="rate",
                        help="check = fast smoke test; rate = useful M5 profiling geometry")
    args = parser.parse_args(argv)
    report = run(args.trace, args.shape)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] == "blocked":
        return 2
    return 0 if report["status"] == "capture_written_replay_not_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
