#!/usr/bin/env python3
"""One-off Phase 0a Metal capture feasibility spike.

This is NOT a benchmark or a Neural Accelerator counter collector.
It validates one existing Precisit FP16 TensorOps kernel, creates one
.gputrace, and leaves Xcode replay/counter inspection to an operator.

Importing this module must not require macOS, MLX or GPU hardware so that
the safety checks can be unit-tested on Linux.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


class PreflightError(RuntimeError):
    """A missing prerequisite or unsafe output path."""


def validate_preflight(system: str, architecture: str,
                       capture_enabled: str | None, trace_path: Path) -> None:
    if system != "Darwin" or architecture not in ("arm64", "aarch64"):
        raise PreflightError("An Apple Silicon Mac is required for the capture spike")
    if capture_enabled != "1":
        raise PreflightError("Set MTL_CAPTURE_ENABLED=1 before starting Python")
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
        raise PreflightError("Full Xcode 27 is required (xcodebuild unavailable)") from exc
    first = result.stdout.splitlines()[0] if result.stdout.strip() else ""
    match = re.match(r"Xcode\s+(\d+)(?:\.|$)", first)
    if not match or int(match.group(1)) < 27:
        raise PreflightError("Full Xcode 27 or newer is required by these probes")
    return result.stdout.strip()


def source_hashes(repo_root: Path) -> dict[str, str]:
    paths = ("tools/phase0a_capture_spike.py", "probes/lane_sweep.py",
             "probes/_common.py")
    return {p: hashlib.sha256((repo_root / p).read_bytes()).hexdigest()
            for p in paths}


def capture_once(mx, dispatch, trace_path: Path) -> None:
    """Capture one evaluated dispatch; always stop once capture was started."""
    mx.metal.start_capture(str(trace_path))
    try:
        mx.eval(dispatch())
    finally:
        mx.metal.stop_capture()


def run(trace_path: Path) -> None:
    validate_preflight(platform.system(), platform.machine(),
                       os.environ.get("MTL_CAPTURE_ENABLED"), trace_path)
    xcode_version()

    # Import only after preflight: lane_sweep imports MLX and applies its
    # existing memory/cache limits. We intentionally inherit those defaults.
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "probes"))
    try:
        import mlx.core as mx
        import numpy as np
        import lane_sweep as ls
    except ImportError as exc:
        raise PreflightError("Install the stock mlx and numpy packages") from exc

    if not hasattr(mx, "metal") or not all(
        callable(getattr(mx.metal, k, None)) for k in ("start_capture", "stop_capture")
    ):
        raise PreflightError("Installed MLX does not expose Metal capture APIs")

    # Use precisely the upstream check-shape, lane descriptor and tile.
    lane = "f16*f16->f32"
    tile = ls.TILES[0]  # 64x128x4
    m, k, n = ls.CHK   # 64x1024x256
    rng = np.random.default_rng([ls.SEED, ls.LANE_INDEX[lane]])
    av, abits = ls.pattern(rng, (m, k), ls.lane_kind(ls.LANES[lane][0]))
    bv, bbits = ls.pattern(rng, (k, n), ls.lane_kind(ls.LANES[lane][2]))
    operands, av, bv = ls.encoded_operands(lane, av, abits, bv, bbits,
                                           None, "lo")
    kernel = ls.build(lane, m, k, n, tile)

    def dispatch():
        return ls.call(kernel, operands, m, n, tile)

    # This also JIT-compiles the kernel and synchronizes a readback.
    out = np.array(dispatch())
    reference = ls.reference(av, bv, None, k)
    error = ls.rel_err(out, reference)
    if not np.isfinite(error) or error >= ls.TOL:
        raise RuntimeError(f"FP16 correctness check failed (relative error {error})")
    del out, reference, av, bv
    mx.eval(dispatch())  # warm-up; not a steady-state assertion

    capture_once(mx, dispatch, trace_path)
    if not trace_path.exists():
        raise RuntimeError("MLX returned from capture without creating .gputrace")

    dev = mx.device_info()
    report = {
        "status": "capture_written_replay_not_verified",
        "lane": lane, "shape": [m, k, n], "tile": list(tile),
        "dispatches_requested": 1,
        "trace_basename": trace_path.name,
        "correctness_relative_error": error,
        "mlx": getattr(mx, "__version__", "unknown"),
        "numpy": np.__version__,
        "xcode": xcode_version().splitlines()[0],
        "device": {k: dev.get(k) for k in ("device_name", "architecture", "memory_size")},
        "sources": source_hashes(root),
        "note": "Open and replay locally in Xcode; hardware counters not collected",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("phase0a-f16.gputrace"),
                        help="new .gputrace bundle path (must not exist)")
    args = parser.parse_args(argv)
    try:
        run(args.trace)
    except (PreflightError, RuntimeError) as exc:
        print(f"Phase 0a blocked/failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
