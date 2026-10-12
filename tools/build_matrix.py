#!/usr/bin/env python3
"""Build (or check) the human-readable views of matrix/capabilities.json.

The JSON is the single source of truth. This tool validates it against
`matrix/schema.json`, cross-checks every cell against the receipt it cites
(lane existence and status, measured rate, baseline, device identity), and
writes:

    matrix/MATRIX.md             grouped tables for reading on GitHub
    matrix/capabilities.csv      flat grid (GitHub renders CSV as a table)

Usage:
    python tools/build_matrix.py            # write the views
    python tools/build_matrix.py --check    # validate + fail if views are stale

Requires `jsonschema` (pip install jsonschema). `--check` is what CI runs on
PRs and on main.
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(ROOT, "matrix", "capabilities.json")
MD = os.path.join(ROOT, "matrix", "MATRIX.md")
CSV = os.path.join(ROOT, "matrix", "capabilities.csv")
SCHEMA = os.path.join(ROOT, "matrix", "schema.json")

PROTOCOL_SHAPE = (2048, 4096, 4096)
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


def schema_errors(cap, root):
    """Validate against matrix/schema.json (jsonschema must be installed)."""
    path = os.path.join(root, "matrix", "schema.json")
    if not os.path.isfile(path):
        return ["schema not found: matrix/schema.json"]
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed (pip install jsonschema)"]
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


def validate(cap, root=ROOT):
    """All structural, schema, receipt-fidelity and identity checks."""
    errors = []
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
                if not os.path.isfile(path):
                    errors.append(f"{prefix}: receipt not found: {cell['receipt']}")
                    continue
                if path not in checked:
                    try:
                        with open(path) as f:
                            rec = json.load(f)
                    except (OSError, ValueError) as exc:
                        errors.append(f"{prefix}: receipt unreadable: {exc}")
                        continue
                    checked[path] = rec
                    errors += receipt_identity_errors(rec, chip, c)
                rec = checked[path]
                if "rows" in rec:  # integer receipt
                    rows = [r for r in rec["rows"] if r.get("lane") == lane]
                    if not rows:
                        errors.append(f"{prefix}: lane not in {cell['receipt']}")
                        continue
                    ok = [r for r in rows if r.get("status") == "ok"
                          and (r.get("M"), r.get("K"), r.get("N")) == PROTOCOL_SHAPE
                          and finite_pos(r.get("gops_best"))]
                    if not ok:
                        errors.append(f"{prefix}: no successful {PROTOCOL_SHAPE} row "
                                      f"in {cell['receipt']}")
                        continue
                    measured = max(r["gops_best"] for r in ok)
                else:  # mapping receipt (float/mixed sweep)
                    lane_rec = rec.get(lane)
                    if not isinstance(lane_rec, dict):
                        errors.append(f"{prefix}: lane not in {cell['receipt']}")
                        continue
                    if lane_rec.get("status") != "ok":
                        errors.append(f"{prefix}: receipt lane status is "
                                      f"{lane_rec.get('status')!r}, not 'ok'")
                        continue
                    rates = lane_rec.get("rates")
                    if not isinstance(rates, dict) or not rates:
                        errors.append(f"{prefix}: receipt lane has no tile rates")
                        continue
                    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               for v in rates.values()):
                        errors.append(f"{prefix}: receipt tile rates must be numbers")
                        continue
                    if not any(finite_pos(v) for v in rates.values()):
                        errors.append(f"{prefix}: receipt lane has no successful tile")
                        continue
                    measured = max(rates.values())
                if abs(rate - measured) > max(0.2, abs(measured) * 1e-6):
                    errors.append(f"{prefix}: gops_best {rate} != receipt's {measured} "
                                  f"in {cell['receipt']}")
                    continue
                base = f16_rate(rec)
                if not finite_pos(base):
                    errors.append(f"{prefix}: receipt has no successful f16*f16->f32 "
                                  f"baseline")
                    continue
                expect = round(rate / base, 3)
                if abs(expect - ratio) > 0.0015:
                    errors.append(f"{prefix}: vs_f16 {ratio} != gops/f16 {expect} "
                                  f"from {cell['receipt']}")
    return errors


def fmt(v):
    return "—" if v is None else f"{v:.2f}×"


def build(cap):
    chips = sorted(cap["chips"], key=chip_key)
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
               "These are lane probes, not model numbers. See [METHODS.md](../METHODS.md).")
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
                elif cell["vs_f16"] >= 2.0:
                    row.append(f"| **{fmt(cell['vs_f16'])}** ")
                else:
                    row.append(f"| {fmt(cell['vs_f16'])} ")
            lines.append("".join(row) + "|")
        lines.append("")
        return lines

    for cls, label in CLASSES:
        if any(cap["chips"][ch].get(cls) for ch in chips):
            out += table(cls)

    out.append("## Machines")
    out.append("")
    out.append("| Key | Device | Architecture | GPU cores | Memory | MLX at measurement |")
    out.append("|---|---|---|---:|---:|---|")
    for k in chips:
        ch = cap["chips"][k]
        out.append(f"| `{k}` | {ch['device_name']} | `{ch['architecture']}` | "
                   f"{ch['gpu_cores']} | {ch['memory_gb']} GB | "
                   f"{ch.get('mlx_at_measurement', '—')} |")
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

    unknown = cap.get("unknown_chips", [])
    if unknown:
        out.append("## Unknown chips")
        out.append("")
        out.append("No measured rows yet: " + ", ".join(f"`{u}`" for u in unknown) + ".")
        out.append("")
    md = "\n".join(out) + "\n"

    csv_lines = ["chip,gpu_cores,class,lane,gops_best,gops_per_core,vs_f16,receipt,evidence"]
    for chip in chips:
        for cls, label in CLASSES:
            for lane, cell in sorted(cap["chips"][chip].get(cls, {}).items()):
                cores = cap["chips"][chip]["gpu_cores"]
                csv_lines.append(
                    f"{chip},{cores},{label.lower()},{lane},{cell['gops_best']:g},"
                    f"{cell['gops_best']/cores:.1f},{cell['vs_f16']},{cell['receipt']},"
                    f"{cap['chips'][chip]['evidence']}")
    csv = "\n".join(csv_lines) + "\n"
    return md, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate and fail if the generated views are stale")
    args = ap.parse_args()

    with open(CAP) as f:
        cap = json.load(f)
    errors = validate(cap)
    if errors:
        print("validation failed:", file=sys.stderr)
        for e in errors:
            print(" ", e, file=sys.stderr)
        return 1
    md, csv = build(cap)

    if args.check:
        stale = []
        for path, want in ((MD, md), (CSV, csv)):
            have = open(path).read() if os.path.isfile(path) else None
            if have != want:
                stale.append(os.path.relpath(path, ROOT))
        if stale:
            print("generated views are stale: " + ", ".join(stale), file=sys.stderr)
            print("run: python tools/build_matrix.py", file=sys.stderr)
            return 1
        print("ok: matrix/capabilities.json valid; views fresh")
        return 0

    with open(MD, "w") as f:
        f.write(md)
    with open(CSV, "w") as f:
        f.write(csv)
    print(f"wrote {os.path.relpath(MD, ROOT)} and {os.path.relpath(CSV, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
