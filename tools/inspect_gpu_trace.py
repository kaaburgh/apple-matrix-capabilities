#!/usr/bin/env python3
"""Experimental local GPU trace diagnostics, NOT a command-buffer verifier.

Xcode 27 gpudebug output has not been validated against real empty and
nonempty captures. Until scrubbed fixtures exist, never infer capture
contents from guessed strings or bundle sizes. Xcode GUI remains the gate.
No bundle data, filenames inside the bundle, or gpudebug text is printed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def bundle_stats(path: Path) -> dict:
    """File count and byte sum only, following no symlinks."""
    if path.is_file() and not path.is_symlink():
        return {"bundle_bytes": path.stat().st_size, "bundle_files": 1}
    total = count = 0
    for root, _, files in os.walk(path, followlinks=False):
        for file in files:
            candidate = Path(root) / file
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
                count += 1
            except OSError:
                pass
    return {"bundle_bytes": total, "bundle_files": count}


def inspect(trace: Path, *, gpudebug: bool = False, timeout: int = 90) -> dict:
    if trace.suffix != ".gputrace" or not trace.exists():
        return {"status": "unverified", "reason": "missing_or_invalid_trace_path"}
    stats = bundle_stats(trace)
    report = {
        "status": "unverified",
        "reason": "command_buffer_and_dispatch_counts_require_xcode",
        **stats,
        "size_is_not_proof_of_commands": True,
    }
    if gpudebug:
        try:
            process = subprocess.run(
                ["xcrun", "gpudebug", "--oneshot", "-t", str(trace), "-c", "list"],
                capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError:
            result = "xcrun_unavailable"
        except subprocess.TimeoutExpired:
            result = "gpudebug_timeout"
        except OSError:
            result = "gpudebug_launch_failed"
        else:
            # NEVER parse or echo arbitrary gpudebug text: previous regexps
            # were invented, CI did not match them, and per-queue lines could
            # be mistaken for totals.
            result = "ran_output_unvalidated" if process.returncode == 0 else "gpudebug_failed"
        report["gpudebug_probe"] = result
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gpudebug", action="store_true",
                        help="probe CLI availability only; no output parsing")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)
    report = inspect(args.trace, gpudebug=args.gpudebug, timeout=args.timeout)
    print(json.dumps(report, sort_keys=True))
    return 2  # always unverified until real fixtures establish semantics


if __name__ == "__main__":
    raise SystemExit(main())
