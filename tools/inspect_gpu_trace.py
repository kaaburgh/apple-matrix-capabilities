#!/usr/bin/env python3
"""Safely inspect whether a local .gputrace contains Metal command buffers.

Uses Xcode 27's gpudebug where available. A missing/unreadable gpudebug
result is explicitly UNVERIFIED, never a successful capture. No raw
trace content, paths or gpudebug output are printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

BUFFER_PATTERNS = (
    re.compile(r"\bSummary:\s*(\d+)\s+command\s+buffers?\b", re.I),
    re.compile(r"\bCommand\s+Buffers\s*:\s*(\d+)\b", re.I),
)
DISPATCH_PATTERN = re.compile(r"\b(?:Dispatch Calls|dispatches)\s*:\s*(\d+)\b", re.I)


def parse_summary(output: str) -> dict:
    buffers = None
    for pattern in BUFFER_PATTERNS:
        match = pattern.search(output)
        if match:
            buffers = int(match.group(1))
            break
    dispatch = DISPATCH_PATTERN.search(output)
    dispatches = int(dispatch.group(1)) if dispatch else None

    if buffers is None:
        return {"status": "unverified", "reason": "command_buffer_count_not_reported"}
    if buffers == 0 or dispatches == 0:
        return {"status": "empty", "command_buffers": buffers,
                "dispatch_calls": dispatches}
    return {"status": "nonempty", "command_buffers": buffers,
            "dispatch_calls": dispatches,
            "note": "Nonempty trace; target kernel identification and counters remain manual"}


def inspect(trace: Path, *, timeout: int = 90) -> dict:
    if not trace.exists() or trace.suffix != ".gputrace":
        return {"status": "unverified", "reason": "missing_or_invalid_trace_path"}
    try:
        process = subprocess.run(
            ["xcrun", "gpudebug", "--oneshot", "-t", str(trace), "-c", "list"],
            capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return {"status": "unverified", "reason": "xcrun_not_found"}
    except subprocess.TimeoutExpired:
        return {"status": "unverified", "reason": "gpudebug_timeout"}
    except OSError:
        return {"status": "unverified", "reason": "gpudebug_launch_failed"}

    if process.returncode != 0:
        return {"status": "unverified", "reason": "gpudebug_failed"}
    return parse_summary(process.stdout + "\n" + process.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)
    status = inspect(args.trace, timeout=args.timeout)
    print(json.dumps(status, sort_keys=True))
    return {"nonempty": 0, "empty": 1, "unverified": 2}[status["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
