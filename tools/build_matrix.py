#!/usr/bin/env python3
"""Build (or check) the human-readable views of matrix/capabilities.json.

The JSON is the single source of truth. This tool:

  * validates it against `matrix/schema.json`;
  * validates **every** receipt under `receipts/` against
    `receipts/schema.json` and the shared semantic rules (finite numbers,
    valid statuses, protocol shape/mode, completion evidence);
  * cross-checks every matrix cell against the receipt it cites: lane
    existence and status, measured rate at the protocol shape, the receipt's
    own `f16*f16->f32` baseline, device/MLX identity, and realpath
    containment inside `receipts/<chip>/`;
  * writes:

        matrix/MATRIX.md             grouped tables for reading on GitHub
        matrix/capabilities.csv      flat grid (GitHub renders CSV as a table)

Completion evidence: a schema-2 receipt written by the current probes carries
`complete: true` after the final write. Receipts written before the flag existed
are accepted when their coverage cross-checks (requested lanes/shapes/tiles vs
the recorded entries) prove the run finished; a checkpoint (`complete: false`)
or a partial coverage cannot back a matrix cell.

Usage:
    python tools/build_matrix.py            # write the views
    python tools/build_matrix.py --check    # validate + fail if views are stale

Requires `jsonschema>=4.18,<5`. `--check` is what CI runs on PRs and on main.
"""

import argparse
import csv
import io
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(ROOT, "matrix", "capabilities.json")
MD = os.path.join(ROOT, "matrix", "MATRIX.md")
CSV = os.path.join(ROOT, "matrix", "capabilities.csv")
SCHEMA = os.path.join(ROOT, "matrix", "schema.json")
RECEIPT_SCHEMA = os.path.join(ROOT, "receipts", "schema.json")

PROTOCOL_SHAPE = (2048, 4096, 4096)
SPREAD_LIMIT = 2.0
CHIP_ORDER = ["m1", "m2", "m3", "m4", "m5-air", "m5-pro", "m5-max", "m5-ultra",
              "m6", "m6-pro", "m6-max"]
CLASSES = [("float_lanes", "Float"), ("mixed_lanes", "Mixed"),
           ("integer_lanes", "Integer")]
TAGS = {"observed", "reported", "inferred", "unknown"}
CHIP_LABELS = {"m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4", "m5-air": "M5 Air",
               "m5-pro": "M5 Pro", "m5-max": "M5 Max", "m5-ultra": "M5 Ultra",
               "m6": "M6", "m6-pro": "M6 Pro", "m6-max": "M6 Max"}


def chip_label(chip):
    return CHIP_LABELS.get(chip, chip)


def chip_key(k):
    return (CHIP_ORDER.index(k) if k in CHIP_ORDER else 99, k)


def finite_pos(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and x > 0)


def finite_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def _reject_const(name):
    raise ValueError(f"non-standard JSON constant not allowed: {name}")


def load_strict(path):
    """json.load that refuses NaN/Infinity literals (strict JSON only)."""
    with open(path) as f:
        return json.load(f, parse_constant=_reject_const)


def is_lane_key(key):
    return isinstance(key, str) and "->" in key


# --- receipt semantics ------------------------------------------------------


def check_errors(where, checks):
    errs = []
    if not isinstance(checks, list):
        return [f"{where}: checks must be an array"]
    for i, c in enumerate(checks):
        if not isinstance(c, dict) or not isinstance(c.get("ok"), bool):
            errs.append(f"{where}: checks[{i}] must carry a boolean ok")
            continue
        for field in ("rel_err", "tolerance"):
            v = c.get(field)
            if isinstance(v, str):  # integer probe uses "exact"
                continue
            if v is not None and not finite_num(v):
                errs.append(f"{where}: checks[{i}].{field} is not finite")
    return errs


def _status_errors(where, entry):
    errs = []
    status = entry.get("status")
    if status not in ("ok", "wrong", "failed"):
        errs.append(f"{where}: status {status!r} not one of ok/wrong/failed")
        return errs
    errs += check_errors(where, entry.get("checks", []))
    if status == "ok":
        if not all(c.get("ok") for c in entry.get("checks", []) if isinstance(c, dict)):
            errs.append(f"{where}: status 'ok' but a recorded check failed")
        if not finite_pos(entry.get("gops_best")):
            errs.append(f"{where}: status 'ok' without a finite positive gops_best")
    rates = entry.get("rates")
    if rates is not None:
        if not isinstance(rates, dict):
            errs.append(f"{where}: rates must be an object")
        else:
            for k, v in rates.items():
                if not finite_pos(v):
                    errs.append(f"{where}: rate {k!r} is not finite and positive")
    samples = entry.get("samples")
    if samples is not None:
        if not isinstance(samples, dict):
            errs.append(f"{where}: samples must be an object")
        else:
            for k, vals in samples.items():
                if not isinstance(vals, list) or not vals or not all(
                        finite_num(v) and v >= 0 for v in vals):
                    errs.append(f"{where}: samples {k!r} must be non-empty "
                                f"arrays of finite non-negative numbers")
    return errs


def int_row_errors(where, row):
    errs = []
    if not isinstance(row.get("lane"), str) or not is_lane_key(row.get("lane")):
        errs.append(f"{where}: row has no lane name")
    tile = row.get("tile")
    if (not isinstance(tile, list) or len(tile) != 3
            or any(not isinstance(v, int) or isinstance(v, bool) for v in tile)):
        errs.append(f"{where}: row tile must be three integers")
    errs += _status_errors(where, row)
    if row.get("status") == "ok":
        for field in ("M", "K", "N"):
            if not (isinstance(row.get(field), int) and not isinstance(row.get(field), bool)
                    and row[field] > 0):
                errs.append(f"{where}: successful row without positive {field}")
        for field in ("ms_best", "ms_med", "gops_best", "gops_med"):
            if row.get(field) is not None and not finite_num(row[field]):
                errs.append(f"{where}: {field} is not finite")
        for i, v in enumerate(row.get("ms_samples", [])):
            if not finite_num(v) or v < 0:
                errs.append(f"{where}: ms_samples[{i}] is not finite/non-negative")
    return errs


def receipt_is_schema2(rec):
    v = rec.get("receipt_schema")
    return isinstance(v, int) and not isinstance(v, bool) and v >= 2


def completion_errors(rec):
    """Prove the receipt ran to completion: explicit flag or coverage cross-check.

    Schema-2 receipts written by the current probes end with `complete: true`.
    Receipts from before the flag exist: their completion must be provable from
    coverage (every requested lane/shape/tile recorded, counts consistent).
    `complete: false` is an interrupted checkpoint and never citable.
    """
    if rec.get("complete") is False:
        return ["incomplete checkpoint (complete=false)"]
    flagged = rec.get("complete") is True
    errs = []  # coverage mismatches are errors even when the flag is set
    if "rows" in rec:  # integer_lanes (rows = lanes x tiles x protocol shapes)
        rows = rec.get("rows")
        res = rec.get("results") or {}
        if not isinstance(rows, list):
            return ["completion unproven: rows is not an array"]
        if res.get("rows_total") != len(rows):
            errs.append("completion unproven: results.rows_total != len(rows)")
        proto = rec.get("protocol") or {}
        lanes = {r.get("lane") for r in rows if isinstance(r, dict)}
        tiles = {tuple(r.get("tile", [])) for r in rows if isinstance(r, dict)}
        shapes = {(r.get("M"), r.get("K"), r.get("N"))
                  for r in rows if isinstance(r, dict)}
        want_shapes = {tuple(int(x) for x in s.split("x"))
                       for s in proto.get("shapes", []) if isinstance(s, str)}
        if want_shapes:
            shapes &= want_shapes
        expected = len(lanes) * len(tiles) * len(shapes)
        seen = {(r.get("lane"), tuple(r.get("tile", [])),
                 (r.get("M"), r.get("K"), r.get("N")))
                for r in rows if isinstance(r, dict)}
        if tiles and shapes and (len(seen) != expected or len(rows) != expected):
            errs.append("completion unproven: rows do not cover "
                        "lanes x tiles x protocol shapes")
    elif "lanes" in rec:  # geometry_sweep
        proto = rec.get("protocol") or {}
        req = proto.get("lanes_requested") or proto.get("lanes")
        labels = proto.get("shapes")
        lanes = rec.get("lanes")
        if not isinstance(lanes, dict) or not req:
            if flagged:
                return errs
            return ["completion unproven: geometry lanes/protocol missing"]
        if sorted(lanes) != sorted(req):
            errs.append("completion unproven: lanes recorded != lanes requested")
        if labels:
            for lane, cells in lanes.items():
                if not isinstance(cells, dict) or sorted(cells) != sorted(labels):
                    errs.append(f"completion unproven: {lane} does not cover "
                                f"every requested shape")
                    break
    else:  # lane_sweep
        proto = rec.get("protocol") or {}
        req = proto.get("lanes_requested")
        present = sorted(k for k in rec if is_lane_key(k))
        res = rec.get("results") or {}
        if not req:
            if not flagged:
                errs.append("completion unproven: no lanes_requested")
        elif present != sorted(req):
            errs.append("completion unproven: lanes recorded != lanes requested")
        if req and res.get("lanes_total") != len(req):
            errs.append("completion unproven: results.lanes_total != lanes_requested")
        counts = sum(res.get(k, 0) for k in ("ok", "wrong", "failed"))
        if res.get("lanes_total") is not None and counts != res.get("lanes_total"):
            errs.append("completion unproven: ok+wrong+failed != lanes_total")
    return errs


def receipt_semantic_errors(rel, rec):
    """Family-specific semantic checks; `rel` is the receipt's relative path."""
    probe = rec.get("probe")
    errs = []
    if probe == "apple-matrix-capabilities/lane_sweep":
        proto = rec.get("protocol") or {}
        if proto and tuple(proto.get("shape", ())) != PROTOCOL_SHAPE:
            errs.append(f"{rel}: protocol.shape must be {PROTOCOL_SHAPE} for "
                        f"cells measured at the standard rate")
        for key, entry in rec.items():
            if is_lane_key(key) and not key.endswith("->f32") and not key.endswith("->i32"):
                errs.append(f"{rel}: lane {key!r} has an unexpected output type")
            if is_lane_key(key):
                errs += _status_errors(f"{rel}:{key}", entry)
    elif probe == "apple-matrix-capabilities/integer_lanes":
        proto = rec.get("protocol") or {}
        shapes = proto.get("shapes")
        want = f"{PROTOCOL_SHAPE[0]}x{PROTOCOL_SHAPE[1]}x{PROTOCOL_SHAPE[2]}"
        if not isinstance(shapes, list) or want not in shapes:
            errs.append(f"{rel}: protocol.shapes must include {want}")
        for i, row in enumerate(rec.get("rows", [])):
            errs += int_row_errors(f"{rel}:rows[{i}]", row)
    elif probe == "apple-matrix-capabilities/geometry_sweep":
        proto = rec.get("protocol") or {}
        lanes = rec.get("lanes")
        if not isinstance(lanes, dict):
            errs.append(f"{rel}: lanes mapping missing")
        else:
            for lane, cells in lanes.items():
                if not isinstance(cells, dict):
                    errs.append(f"{rel}:{lane}: cells must be an object")
                    continue
                for label, cell in cells.items():
                    errs += _status_errors(f"{rel}:{lane}:{label}", cell)
        if proto.get("shapes") and not isinstance(proto["shapes"], dict):
            errs.append(f"{rel}: protocol.shapes must be an object")
    if (probe in ("apple-matrix-capabilities/lane_sweep",
                  "apple-matrix-capabilities/integer_lanes",
                  "apple-matrix-capabilities/geometry_sweep")
            and receipt_is_schema2(rec)):
        errs += [f"{rel}: {e}" for e in completion_errors(rec)]
    return errs


def receipt_schema_errors(rel, rec, root=ROOT):
    path = os.path.join(root, "receipts", "schema.json")
    if not os.path.isfile(path):
        return [f"{rel}: receipts/schema.json not found"]
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed (pip install 'jsonschema>=4.18,<5')"]
    with open(path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(rec, schema)
    except jsonschema.ValidationError as exc:
        loc = "/".join(str(p) for p in exc.absolute_path)
        return [f"{rel}: schema: {loc or '<root>'}: {exc.message}"]
    return []


def scan_receipts(root=ROOT):
    """Validate every receipt under receipts/. Returns (errors, cache)."""
    errors, cache = [], {}
    base = os.path.join(root, "receipts")
    if not os.path.isdir(base):
        return ["receipts/ is missing"], cache
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith(".json") or name == "schema.json":
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            try:
                rec = load_strict(path)
            except (OSError, ValueError) as exc:
                errors.append(f"{rel}: unreadable or non-strict JSON: {exc}")
                continue
            cache[rel.replace("\\", "/")] = rec
            errors += receipt_schema_errors(rel, rec, root)
            errors += receipt_semantic_errors(rel, rec)
    return errors, cache


# --- capability matrix validation -------------------------------------------


def schema_errors(cap, root):
    """Validate capabilities.json against matrix/schema.json."""
    path = os.path.join(root, "matrix", "schema.json")
    if not os.path.isfile(path):
        return ["schema not found: matrix/schema.json"]
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed (pip install 'jsonschema>=4.18,<5')"]
    with open(path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(cap, schema)
    except jsonschema.ValidationError as exc:
        loc = "/".join(str(p) for p in exc.absolute_path)
        return [f"schema: {loc or '<root>'}: {exc.message}"]
    return []


def f16_rate(receipt):
    """The successful f16*f16->f32 rate in a receipt at the protocol shape."""
    if "rows" in receipt:  # integer receipt: rows carry M/K/N and status
        best = None
        for r in receipt["rows"]:
            if (r.get("lane") == "f16*f16->f32" and r.get("status") == "ok"
                    and (r.get("M"), r.get("K"), r.get("N")) == PROTOCOL_SHAPE
                    and finite_pos(r.get("gops_best"))):
                best = r["gops_best"] if best is None else max(best, r["gops_best"])
        return best
    lane = receipt.get("f16*f16->f32") or {}
    if lane.get("status") != "ok" or not finite_pos(lane.get("gops_best")):
        return None
    return lane["gops_best"]


def receipt_identity_errors(rec, chip, c):
    """The receipt's device report must match the chip entry it backs."""
    dev = rec.get("device")
    if not isinstance(dev, dict):
        return [f"{chip}: receipt has no device report"]
    errs = []
    if dev.get("device_name") != c.get("device_name"):
        errs.append(f"{chip}: receipt device_name {dev.get('device_name')!r} != "
                    f"{c.get('device_name')!r}")
    if dev.get("architecture") != c.get("architecture"):
        errs.append(f"{chip}: receipt architecture {dev.get('architecture')!r} != "
                    f"{c.get('architecture')!r}")
    mem = dev.get("memory_size")
    if isinstance(mem, (int, float)) and not isinstance(mem, bool):
        if int(mem) // 2**30 != c.get("memory_gb"):
            errs.append(f"{chip}: receipt memory_size {mem} != "
                        f"memory_gb {c.get('memory_gb')!r}")
    mlx = rec.get("mlx")
    if c.get("mlx_at_measurement") and mlx != c.get("mlx_at_measurement"):
        errs.append(f"{chip}: receipt mlx {mlx!r} != "
                    f"mlx_at_measurement {c.get('mlx_at_measurement')!r}")
    return errs


def _cell_receipt_errors(prefix, rel, chip, root):
    """Path containment (incl. symlinks) for a cited receipt."""
    parts = rel.split("/")
    lexical = os.path.join(root, *parts)
    if not os.path.isfile(lexical):
        return [f"{prefix}: receipt not found: {rel}"]
    root_real = os.path.realpath(root)
    chip_dir_real = os.path.realpath(os.path.join(root, "receipts", chip))
    real = os.path.realpath(lexical)
    if (not (real == chip_dir_real or real.startswith(chip_dir_real + os.sep))
            or not real.startswith(root_real + os.sep)):
        return [f"{prefix}: receipt escapes receipts/{chip} (symlink?): {rel}"]
    return []


def _cell_rate_from_receipt(prefix, lane, rec, rel):
    """Measured best rate for `lane` at the protocol shape, or errors."""
    errs = []
    if "rows" in rec:  # integer receipt
        rows = [r for r in rec.get("rows", [])
                if isinstance(r, dict) and r.get("lane") == lane]
        if not rows:
            return None, [f"{prefix}: lane not in {rel}"]
        ok = [r for r in rows if r.get("status") == "ok"
              and (r.get("M"), r.get("K"), r.get("N")) == PROTOCOL_SHAPE
              and finite_pos(r.get("gops_best"))]
        if not ok:
            return None, [f"{prefix}: no successful {PROTOCOL_SHAPE} row in {rel}"]
        return max(r["gops_best"] for r in ok), errs
    lane_rec = rec.get(lane)
    if not isinstance(lane_rec, dict):
        return None, [f"{prefix}: lane not in {rel}"]
    if lane_rec.get("status") != "ok":
        return None, [f"{prefix}: receipt lane status is "
                      f"{lane_rec.get('status')!r}, not 'ok'"]
    rates = lane_rec.get("rates")
    if not isinstance(rates, dict) or not rates:
        return None, [f"{prefix}: receipt lane has no tile rates"]
    if not all(finite_num(v) for v in rates.values()):
        return None, [f"{prefix}: receipt tile rates must be finite numbers"]
    if not any(finite_pos(v) for v in rates.values()):
        return None, [f"{prefix}: receipt lane has no successful tile"]
    return max(rates.values()), errs


def validate(cap, root=ROOT):
    """All structural, schema, receipt-fidelity and identity checks.

    Returns a list of error strings (empty = valid). Validates every receipt in
    receipts/ as well as the cells of `cap`.
    """
    errors, receipts = scan_receipts(root)
    errors += schema_errors(cap, root)
    chips = cap.get("chips")
    if not isinstance(chips, dict) or not chips:
        errors.append("chips missing")
        return errors
    unknown = cap.get("unknown_chips", [])
    if isinstance(unknown, list):
        for u in unknown:
            if u in chips:
                errors.append(f"unknown_chips contains measured chip {u!r}")
    checked = {}
    for chip, c in chips.items():
        if not isinstance(c, dict):
            errors.append(f"{chip}: must be an object")
            continue
        for field in ("device_name", "architecture", "gpu_cores", "memory_gb", "evidence"):
            if field not in c:
                errors.append(f"{chip}: missing {field}")
        for field in ("device_name", "architecture"):
            if not (isinstance(c.get(field), str) and c.get(field)):
                errors.append(f"{chip}: {field} must be a non-empty string")
        gc = c.get("gpu_cores")
        if not (isinstance(gc, int) and not isinstance(gc, bool) and gc > 0):
            errors.append(f"{chip}: gpu_cores must be a positive integer")
        if c.get("evidence") not in TAGS - {"unknown"}:
            errors.append(f"{chip}: evidence must be one of {sorted(TAGS - {'unknown'})}")
        has_cells = any(c.get(cls) for cls, _ in CLASSES)
        if has_cells and c.get("evidence") != "observed":
            errors.append(f"{chip}: evidence {c.get('evidence')!r} cannot carry "
                          "measured lanes; the matrix currently represents "
                          "observed (receipt-backed) rows only")
        for cls, _ in CLASSES:
            lanes = c.get(cls, {})
            if not isinstance(lanes, dict):
                errors.append(f"{chip}.{cls}: must be an object")
                continue
            for lane, cell in lanes.items():
                prefix = f"{chip}.{cls}.{lane}"
                if not isinstance(cell, dict):
                    errors.append(f"{prefix}: must be an object")
                    continue
                missing = [f for f in ("gops_best", "vs_f16", "receipt") if f not in cell]
                if missing:
                    errors.append(f"{prefix}: missing {', '.join(missing)}")
                    continue
                rate, ratio = cell["gops_best"], cell["vs_f16"]
                if not finite_pos(rate):
                    errors.append(f"{prefix}: gops_best must be finite and positive")
                if not finite_pos(ratio):
                    errors.append(f"{prefix}: vs_f16 must be finite and positive")
                if not finite_pos(rate) or not finite_pos(ratio):
                    continue
                rel = cell["receipt"]
                if not isinstance(rel, str):
                    errors.append(f"{prefix}: receipt must be a string")
                    continue
                rel = rel.replace("\\", "/")
                parts = rel.split("/")
                if (os.path.isabs(rel) or ".." in parts or len(parts) < 3
                        or parts[0] != "receipts" or parts[1] != chip):
                    errors.append(f"{prefix}: receipt must be receipts/{chip}/<file>, "
                                  f"got {cell['receipt']!r}")
                    continue
                path = os.path.join(root, *parts)
                errs = _cell_receipt_errors(prefix, rel, chip, root)
                if errs:
                    errors += errs
                    continue
                if path not in checked:
                    rec = receipts.get(rel) if root == ROOT else None
                    if rec is None:
                        try:
                            rec = load_strict(path)
                        except (OSError, ValueError) as exc:
                            errors.append(f"{prefix}: receipt unreadable: {exc}")
                            continue
                    checked[path] = rec
                    errors += receipt_identity_errors(rec, chip, c)
                rec = checked[path]
                if not receipt_is_schema2(rec):
                    errors.append(f"{prefix}: cited receipt predates schema 2; "
                                  f"completion cannot be proven ({rel})")
                    continue
                completion = completion_errors(rec)
                if completion:
                    errors.append(f"{prefix}: {completion[0]} ({rel})")
                    continue
                measured, errs = _cell_rate_from_receipt(prefix, lane, rec, rel)
                if errs:
                    errors += errs
                    continue
                if abs(rate - measured) > max(0.2, abs(measured) * 1e-6):
                    errors.append(f"{prefix}: gops_best {rate} != receipt's {measured} "
                                  f"(rates are rounded to 1 decimal, hence the 0.2 "
                                  f"tolerance) in {rel}")
                    continue
                base = f16_rate(rec)
                if not finite_pos(base):
                    errors.append(f"{prefix}: receipt has no successful f16*f16->f32 "
                                  f"baseline")
                    continue
                expect = round(rate / base, 3)
                if abs(expect - ratio) > 0.0015:
                    errors.append(f"{prefix}: vs_f16 {ratio} != gops/f16 {expect} "
                                  f"(vs_f16 is rounded to 3 decimals) from {rel}")
    return errors


# --- views ------------------------------------------------------------------


def sample_spread(values):
    """max/min of a list of positive samples, or None."""
    vals = [v for v in values if finite_pos(v)]
    if len(vals) < 2:
        return None
    return max(vals) / min(vals)


def lane_spread(rec, lane):
    """Worst (max/min) sample spread for `lane` in a receipt, or None."""
    spreads = []
    if "rows" in rec:
        for r in rec.get("rows", []):
            if isinstance(r, dict) and r.get("lane") == lane:
                s = sample_spread(r.get("ms_samples", []))
                if s:
                    spreads.append(s)
    elif "lanes" in rec:
        for cell in (rec.get("lanes", {}).get(lane) or {}).values():
            for vals in (cell.get("samples") or {}).values():
                s = sample_spread(vals)
                if s:
                    spreads.append(s)
    else:
        entry = rec.get(lane) or {}
        for vals in (entry.get("samples") or {}).values():
            s = sample_spread(vals)
            if s:
                spreads.append(s)
    return max(spreads) if spreads else None


def cited_by(cap):
    """{receipt relpath: [(chip, cls, lane), ...]} in deterministic order."""
    out = {}
    chips = cap.get("chips")
    for chip in sorted(chips, key=chip_key):
        for cls, _ in CLASSES:
            for lane in sorted(chips[chip].get(cls, {})):
                rel = chips[chip][cls][lane]["receipt"].replace("\\", "/")
                out.setdefault(rel, []).append((chip, cls, lane))
    return out


def baseline_info(cap, root=ROOT):
    """{chip: (f16 GOPS, receipt relpath)} from the receipts the chip cites."""
    out = {}
    for chip, c in cap.get("chips", {}).items():
        rels = sorted({c[cls][lane]["receipt"].replace("\\", "/")
                       for cls, _ in CLASSES for lane in c.get(cls, {})})
        for rel in rels:
            try:
                rec = load_strict(os.path.join(root, *rel.split("/")))
            except (OSError, ValueError):
                continue
            rate = f16_rate(rec)
            if finite_pos(rate):
                out[chip] = (rate, rel)
                break
    return out


def fmt(v):
    return "—" if v is None else f"{v:.2f}×"


def build(cap, root=ROOT):
    chips = sorted(cap["chips"], key=chip_key)
    cited = cited_by(cap)
    flags = set()  # (receipt, lane) whose samples spread > SPREAD_LIMIT
    for rel, cells in cited.items():
        try:
            rec = load_strict(os.path.join(root, *rel.split("/")))
        except (OSError, ValueError):
            continue
        for _, _, lane in cells:
            s = lane_spread(rec, lane)
            if s is not None and s > SPREAD_LIMIT:
                flags.add((rel, lane))
    baselines = baseline_info(cap, root)

    def flagged(receipt, lane):
        return (receipt.replace("\\", "/"), lane) in flags

    out = []
    out.append("# Capability matrix (generated)")
    out.append("")
    out.append("> Generated from [`capabilities.json`](capabilities.json) by "
               "[`tools/build_matrix.py`](../tools/build_matrix.py). "
               "Do not edit by hand; edit the JSON and run the tool.")
    out.append("")
    out.append("Rates are GOPS measured at 2048×4096×4096 (`mode::multiply`), the best result "
               "the receipt records: the standard sweep tries five tile configurations and "
               "each receipt records the tiles its lanes actually ran. `vs f16` divides the "
               "same receipt's `f16*f16->f32` rate, so each chip is compared only with itself. "
               "A `⚠` marks a cell whose cited receipt shows a max/min timing spread above "
               f"{SPREAD_LIMIT:.0f}× across samples: treat it as unstable and rerun before "
               "building any dispatch rule on it. These are lane probes, not model numbers. "
               "See [METHODS.md](../METHODS.md) and [README.md](../README.md) for the "
               "evidence-status model (only observed, receipt-backed rows are published).")
    out.append("")

    def header():
        return "| Lane | " + " | ".join(
            f"{chip_label(k)} ({cap['chips'][k]['gpu_cores']}c)" for k in chips) + " |"

    def table(cls):
        lines = [f"## {label} lanes", "",
                 header(),
                 "|---|" + "|".join("--:" for _ in chips) + "|"]
        all_lanes = []
        for chip in chips:
            for lane in cap["chips"][chip].get(cls, {}):
                if lane not in all_lanes:
                    all_lanes.append(lane)
        all_lanes.sort()
        for lane in all_lanes:
            row = [f"| `{lane}` "]
            for chip in chips:
                cell = cap["chips"][chip].get(cls, {}).get(lane)
                if not cell:
                    row.append("| — ")
                else:
                    mark = " ⚠" if flagged(cell["receipt"], lane) else ""
                    val = fmt(cell["vs_f16"]) + mark
                    row.append(f"| **{val}** " if cell["vs_f16"] >= 2.0 else f"| {val} ")
            lines.append("".join(row) + "|")
        lines.append("")
        return lines

    for cls, label in CLASSES:
        if any(cap["chips"][ch].get(cls) for ch in chips):
            out += table(cls)

    out.append("## Machines")
    out.append("")
    out.append("| Key | Device | Architecture | GPU cores | Memory | f16 baseline (GOPS) | "
               "MLX at measurement |")
    out.append("|---|---|---|---:|---:|---:|---|")
    for k in chips:
        ch = cap["chips"][k]
        base = baselines.get(k)
        base_cell = f"{base[0]:,.0f}" if base else "—"
        out.append(f"| `{k}` | {ch['device_name']} | `{ch['architecture']}` | "
                   f"{ch['gpu_cores']} | {ch['memory_gb']} GB | {base_cell} | "
                   f"{ch.get('mlx_at_measurement', '—')} |")
    out.append("")
    notes = [(k, cap["chips"][k].get("note")) for k in chips
             if cap["chips"][k].get("note")]
    if notes:
        out.append("Chip notes:")
        out.append("")
        for k, note in notes:
            out.append(f"- `{k}`: {note}")
        out.append("")
    base_lines = [f"`{k}` {baselines[k][0]:,.0f} GOPS from `{baselines[k][1]}`"
                  for k in chips if k in baselines]
    if base_lines:
        out.append("f16 denominator per chip (from the receipt the chip's cells cite, "
                   "so the ratio cancels chip/clock differences): "
                   + "; ".join(base_lines) + ".")
        out.append("")
    out.append("Absolute rates are whole-GPU for the measured SKU; divide by the core count for "
               "an approximate per-core figure. The `vs f16` ratios are same-chip and cancel the "
               "core count.")
    out.append("")
    out.append("## Absolute rates (GOPS best)")
    out.append("")
    out.append(header())
    out.append("|---|" + "|".join("--:" for _ in chips) + "|")
    all_lanes = []
    for chip in chips:
        for cls, _ in CLASSES:
            for lane in cap["chips"][chip].get(cls, {}):
                if lane not in all_lanes:
                    all_lanes.append(lane)
    for lane in sorted(all_lanes):
        row = [f"| `{lane}` "]
        for chip in chips:
            val = None
            for cls, _ in CLASSES:
                cell = cap["chips"][chip].get(cls, {}).get(lane)
                if cell:
                    val = cell["gops_best"]
            row.append("| — " if val is None else f"| {val:,.0f} ")
        out.append("".join(row) + "|")
    out.append("")

    out.append("## Features (observed)")
    out.append("")
    out.append("| Chip | " + " | ".join(
        sorted({k for ch in chips for k in cap["chips"][ch].get("features", {})})) + " |")
    keys = sorted({k for ch in chips for k in cap["chips"][ch].get("features", {})})
    out.append("|---|" + "|".join("---" for _ in keys) + "|")
    for chip in chips:
        feats = cap["chips"][chip].get("features", {})
        out.append(f"| {chip} | " + " | ".join(
            ("yes" if feats.get(k) else "no") if k in feats else "—" for k in keys) + " |")
    out.append("")

    out.append("## Receipts")
    out.append("")
    out.append("Every receipt in `receipts/` is schema-checked in CI. "
               "`complete` is the receipt's own completion evidence (`flag`) or "
               "coverage-proven for receipts written before the flag existed "
               "(`coverage`); legacy/imported evidence receipts cannot back cells.")
    out.append("")
    out.append("| Receipt | Probe | Complete | Cited cells | Max sample spread |")
    out.append("|---|---|---|---:|---:|")
    all_receipts = []
    base = os.path.join(root, "receipts")
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".json") and name != "schema.json":
                rel = os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/")
                all_receipts.append(rel)
    for rel in sorted(all_receipts):
        try:
            rec = load_strict(os.path.join(root, *rel.split("/")))
        except (OSError, ValueError):
            out.append(f"| `{rel}` | — | unreadable | 0 | — |")
            continue
        probe = rec.get("probe") or "legacy (pre-schema)"
        cited_cells = cited.get(rel, [])
        n = len(cited_cells)
        if rec.get("complete") is True:
            comp = "flag"
        elif receipt_is_schema2(rec) and not completion_errors(rec):
            comp = "coverage"
        else:
            comp = "legacy" if not receipt_is_schema2(rec) else "unproven"
        spreads = [s for _, _, lane in cited_cells
                   if (s := lane_spread(rec, lane)) is not None]
        max_spread = f"{max(spreads):.2f}×" + (" ⚠" if max(spreads) > SPREAD_LIMIT else "") \
            if spreads else "—"
        out.append(f"| `{rel}` | `{probe}` | {comp} | {n} | {max_spread} |")
    out.append("")

    unknown = cap.get("unknown_chips", [])
    if unknown:
        out.append("## Unknown chips")
        out.append("")
        out.append("No measured rows yet: " + ", ".join(f"`{u}`" for u in unknown) + ".")
        out.append("")
    md = "\n".join(out) + "\n"

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["chip", "gpu_cores", "class", "lane", "gops_best", "gops_per_core",
                "vs_f16", "spread_flag", "receipt", "evidence"])
    for chip in chips:
        for cls, label in CLASSES:
            for lane, cell in sorted(cap["chips"][chip].get(cls, {}).items()):
                cores = cap["chips"][chip]["gpu_cores"]
                w.writerow([chip, cores, label.lower(), lane, f"{cell['gops_best']:g}",
                            f"{cell['gops_best'] / cores:.1f}", cell["vs_f16"],
                            "spread>2x" if flagged(cell["receipt"], lane) else "",
                            cell["receipt"], cap["chips"][chip]["evidence"]])
    return md, buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate and fail if the generated views are stale")
    args = ap.parse_args()

    try:
        cap = load_strict(CAP)
    except (OSError, ValueError) as exc:
        print(f"matrix/capabilities.json unreadable: {exc}", file=sys.stderr)
        return 1
    errors = validate(cap)
    if errors:
        print("validation failed:", file=sys.stderr)
        for e in errors:
            print(" ", e, file=sys.stderr)
        return 1
    md, csv_text = build(cap)

    if args.check:
        stale = []
        for path, want in ((MD, md), (CSV, csv_text)):
            have = open(path).read() if os.path.isfile(path) else None
            if have != want:
                stale.append(os.path.relpath(path, ROOT))
        if stale:
            print("generated views are stale: " + ", ".join(stale), file=sys.stderr)
            print("run: python tools/build_matrix.py", file=sys.stderr)
            return 1
        print("ok: matrix/capabilities.json valid; every receipt validated; views fresh")
        return 0

    with open(MD, "w") as f:
        f.write(md)
    with open(CSV, "w") as f:
        f.write(csv_text)
    print(f"wrote {os.path.relpath(MD, ROOT)} and {os.path.relpath(CSV, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
