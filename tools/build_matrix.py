#!/usr/bin/env python3
"""Build (or check) the human-readable views of matrix/capabilities.json.

The JSON is the single source of truth. This tool validates it, cross-checks
every vs_f16 against the receipt it cites, and writes:

    matrix/MATRIX.md             grouped tables for reading on GitHub
    matrix/capabilities.csv      flat grid (GitHub renders CSV as a table)

Usage:
    python tools/build_matrix.py            # write the views
    python tools/build_matrix.py --check    # validate + fail if views are stale

No third-party dependencies. `--check` is what CI runs on PRs and on main.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(ROOT, "matrix", "capabilities.json")
MD = os.path.join(ROOT, "matrix", "MATRIX.md")
CSV = os.path.join(ROOT, "matrix", "capabilities.csv")

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


def f16_rate(receipt):
    """The f16*f16->f32 best rate in a receipt, float or integer shapes."""
    if "rows" in receipt:  # integer receipt: rows at M=2048 K=4096 N=4096
        best = None
        for r in receipt["rows"]:
            if (r.get("lane") == "f16*f16->f32" and r.get("status") == "ok"
                    and r.get("M") == 2048 and r.get("K") == 4096
                    and r.get("N") == 4096):
                best = r["gops_best"] if best is None else max(best, r["gops_best"])
        return best
    lane = receipt.get("f16*f16->f32", {})
    return lane.get("gops_best")


def validate(cap):
    errors = []
    if cap.get("schema") != 1:
        errors.append("schema must be 1")
    if not isinstance(cap.get("chips"), dict) or not cap["chips"]:
        errors.append("chips missing")
        return errors
    for chip, c in cap["chips"].items():
        for field in ("device_name", "architecture", "memory_gb", "evidence"):
            if field not in c:
                errors.append(f"{chip}: missing {field}")
        if c.get("evidence") not in TAGS - {"unknown"}:
            errors.append(f"{chip}: evidence must be one of {sorted(TAGS - {'unknown'})}")
        for cls, _ in CLASSES:
            lanes = c.get(cls, {})
            if not isinstance(lanes, dict):
                errors.append(f"{chip}.{cls}: must be an object")
                continue
            for lane, cell in lanes.items():
                for field in ("gops_best", "vs_f16", "receipt"):
                    if field not in cell:
                        errors.append(f"{chip}.{cls}.{lane}: missing {field}")
                path = os.path.join(ROOT, cell.get("receipt", ""))
                if not os.path.isfile(path):
                    errors.append(f"{chip}.{cls}.{lane}: receipt not found: {cell.get('receipt')}")
                    continue
                with open(path) as f:
                    rec = json.load(f)
                base = f16_rate(rec)
                if not base:
                    errors.append(f"{chip}.{cls}.{lane}: receipt has no f16 x f16 lane")
                    continue
                expect = round(cell["gops_best"] / base, 3)
                if abs(expect - cell["vs_f16"]) > 0.0015:
                    errors.append(
                        f"{chip}.{cls}.{lane}: vs_f16 {cell['vs_f16']} != "
                        f"gops/f16 {expect} from {cell['receipt']}")
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
    out.append("Rates are GOPS measured at 2048×4096×4096 (best of five tiles, "
               "`mode::multiply`); `vs f16` divides the same receipt's `f16*f16` rate, "
               "so each chip is compared only with itself. These are lane probes, "
               "not model numbers. See [METHODS.md](../METHODS.md).")
    out.append("")

    def table(cls):
        lines = [f"## {label} lanes", "",
                 "| Lane | " + " | ".join(chip_label(c) for c in chips) + " |",
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

    out.append("## Absolute rates (GOPS best)")
    out.append("")
    out.append("| Lane | " + " | ".join(chip_label(c) for c in chips) + " |")
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

    csv_lines = ["chip,class,lane,gops_best,vs_f16,receipt,evidence"]
    for chip in chips:
        for cls, label in CLASSES:
            for lane, cell in sorted(cap["chips"][chip].get(cls, {}).items()):
                csv_lines.append(
                    f"{chip},{label.lower()},{lane},{cell['gops_best']:g},"
                    f"{cell['vs_f16']},{cell['receipt']},{cap['chips'][chip]['evidence']}")
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
