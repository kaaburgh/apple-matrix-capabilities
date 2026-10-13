#!/usr/bin/env python3
"""Shared helpers for the capability probes.

One implementation of the pieces the probes must agree on: strict JSON for
receipts (no NaN/Infinity), atomic receipt writes, error sanitization, source
hashes, toolchain/device metadata collection and CLI argument validation.
Imported by `lane_sweep.py`, `integer_lanes.py` and `geometry_sweep.py`; each
receipt records the hashes of every file it depends on (`sources`).
"""

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile

# --- strict JSON -----------------------------------------------------------


def _reject_const(name):
    raise ValueError(f"non-standard JSON constant not allowed: {name}")


def load_json(path):
    """json.load that refuses NaN/Infinity (strict JSON only)."""
    with open(path) as f:
        return json.load(f, parse_constant=_reject_const)


def write_receipt(out, path):
    """Atomically replace `path` with `out` as strict JSON.

    Writes to a temporary sibling and renames it into place, so an interrupted
    or failing write cannot destroy the previous valid checkpoint. Refuses to
    serialize NaN/Infinity.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".receipt-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, indent=1, allow_nan=False, default=str)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- error sanitization ----------------------------------------------------

_PATH_RE = re.compile(r"(?:/[\w.\-+@]+){2,}/?|~/[\w.\-+@/]*")
_HOST_RE = re.compile(r"\b[\w-]+\.(?:local|internal|lan|home|corp|intra)\b")
_KEYVAL_RE = re.compile(
    r"(?i)\b(pass(?:word)?|secret|token|api[_-]?key|credential)s?\s*[=:]\s*\S+")


def sanitize(text, limit=200):
    """One safe summary line for a public receipt.

    Handles None and exceptions whose str() is empty (no IndexError), strips
    path-like tokens generally (not only selected prefixes), redacts
    credential-like key=value text and generic private host suffixes. Detailed
    raw diagnostics stay in the contributor's local run log.
    """
    s = "" if text is None else str(text)
    lines = s.strip().splitlines()
    line = lines[0].strip() if lines else ""
    line = _PATH_RE.sub("<path>", line)
    line = _HOST_RE.sub("<host>", line)
    line = _KEYVAL_RE.sub(r"\1=<redacted>", line)
    return line[:limit]


_CATEGORY_BY_TYPE = {
    "MemoryError": "memory",
    "TimeoutError": "timeout",
    "FileNotFoundError": "file",
    "PermissionError": "permission",
    "IsADirectoryError": "file",
    "NotADirectoryError": "file",
    "OSError": "os",
    "ValueError": "value",
    "OverflowError": "value",
    "ZeroDivisionError": "value",
    "TypeError": "type",
    "KeyError": "lookup",
    "IndexError": "lookup",
    "AttributeError": "attribute",
    "RuntimeError": "runtime",
    "AssertionError": "assertion",
    "PackingFailed": "packing",
}


def classify(exc):
    """A controlled category for an exception; never carries free text."""
    return _CATEGORY_BY_TYPE.get(type(exc).__name__, "other")


def err_obj(exc, phase, version=3):
    """Structured failure for a receipt.

    Schema 3 receipts carry only a controlled category plus the exception
    type: arbitrary exception text is not safe to publish (it can embed
    paths, hostnames or credentials). Schema-1/2 historical receipts keep the
    sanitized first line; the probes print raw diagnostics to the local
    console instead.
    """
    obj = {"phase": phase, "type": type(exc).__name__, "category": classify(exc)}
    if version < 3:
        obj["message"] = sanitize(exc)
    return obj


class PackingFailed(RuntimeError):
    """Neither nibble order passed; carries the orders actually attempted."""

    def __init__(self, attempts, cause=None):
        super().__init__(f"packing discovery failed after {list(attempts)}")
        self.attempts = list(attempts)
        self.cause = cause


def finite_check(chk, ensure=False):
    """Make one check entry JSON-safe: non-finite numbers become null + flag.

    A GPU result can be non-finite (NaN/Inf), which is a real numerical
    outcome that must be reported, not an abort: the check stays `ok: false`
    with `rel_err: null` and `non_finite: true`. `ensure` forces the
    diagnostic fields into existence for new (schema 3) receipts.
    """
    out = dict(chk)
    if ensure:
        out.setdefault("rel_err", 0.0)
        out.setdefault("tolerance", None)
        out.setdefault("ok", False)
    for k in ("rel_err", "tolerance"):
        v = out.get(k)
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
            out["non_finite"] = True
    return out


# --- provenance ------------------------------------------------------------


def source_hashes(paths):
    """{basename: sha256} for every file a probe depends on."""
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            out[os.path.basename(p)] = hashlib.sha256(f.read()).hexdigest()
    return out


def probe_files(*extra):
    """This probe's own file plus shared helpers, for `source_hashes`."""
    here = os.path.dirname(os.path.abspath(__file__))
    return list(extra) + [os.path.join(here, "_common.py")]


# --- machine metadata ------------------------------------------------------


def run_text(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout or ""
    except Exception:
        return ""


def first_line(text, default="unknown"):
    lines = text.strip().splitlines()
    return lines[0] if lines else default


def toolchain_meta():
    xc = run_text(["xcrun", "xcodebuild", "-version"]) or run_text(
        ["xcodebuild", "-version"])
    lines = xc.strip().splitlines()
    return {
        "xcode": lines[0] if lines else "unknown",
        "xcode_build": lines[1] if len(lines) > 1 else "unknown",
        "sdk": first_line(run_text(["xcrun", "--sdk", "macosx",
                                    "--show-sdk-version"])),
        "metal": first_line(run_text(["xcrun", "metal", "--version"])),
    }


def gpu_cores():
    """GPU core count from system_profiler, or 'unknown' (never guessed)."""
    out = run_text(["system_profiler", "SPDisplaysDataType"], timeout=20)
    m = re.search(r"Total Number of Cores:\s*(\d+)", out)
    return int(m.group(1)) if m else "unknown"


def power_meta():
    power = {"source": "unknown", "low_power_mode": "unknown"}
    batt = run_text(["pmset", "-g", "batt"])
    power["source"] = ("ac" if "AC Power" in batt
                       else "battery" if "Battery Power" in batt else "unknown")
    m = re.search(r"lowpowermode\s+(\d)", run_text(["pmset", "-g"]))
    if m:
        power["low_power_mode"] = (m.group(1) == "1")
    return power


# --- CLI argument validation ----------------------------------------------


def parse_int_tuple(s, name, n=3, allow_zero=False):
    parts = s.split("x")
    if len(parts) != n:
        raise ValueError(f"{name}: expected {n} 'x'-separated positive integers, "
                         f"got {s!r}")
    vals = []
    for p in parts:
        if not p.isdigit() or (not allow_zero and int(p) <= 0):
            raise ValueError(f"{name}: components must be "
                             f"{'non-negative' if allow_zero else 'positive'} "
                             f"integers, got {s!r}")
        vals.append(int(p))
    return tuple(vals)


def validate_tile(t, name="tile"):
    if len(t) != 3 or any(not isinstance(v, int) for v in t):
        raise ValueError(f"{name}: expected three integers (rows x cols x "
                         f"simdgroups), got {t!r}")
    mt, nt, sg = t
    if mt <= 0 or nt <= 0:
        raise ValueError(f"{name}: rows and cols must be positive, got {t!r}")
    if sg not in (1, 2, 4, 8):
        raise ValueError(f"{name}: simdgroups must be one of 1/2/4/8, got {sg}")
    return (mt, nt, sg)
