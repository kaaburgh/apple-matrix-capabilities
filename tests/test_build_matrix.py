#!/usr/bin/env python3
"""Regression tests for the receipt-backed validation in tools/build_matrix.py.

Run from the repository root:

    python -m unittest discover -s tests -v

Fixtures are internally consistent (rates derived from samples exactly as the
probes derive them), so every validation rule can be exercised. No hardware and
no real receipts are needed; jsonschema must be installed.
"""

import importlib.util
import json
import os
import shutil
import statistics
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "build_matrix", os.path.join(ROOT, "tools", "build_matrix.py"))
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

SHAPE = (2048, 4096, 4096)
CONST = 2.0 * SHAPE[0] * SHAPE[1] * SHAPE[2] / 1e6  # rate = CONST / min_ms
TILES = ["64x128x4", "64x64x2"]


def samples_for(rate, reps=5):
    """A plausible sample list whose min gives approximately `rate` GOPS."""
    mmin = CONST / rate
    return [round(mmin * (1 + 0.01 * i), 3) for i in range(reps)]


def lane_from_samples(samples, relax=False, dims=SHAPE, packing=None,
                      attempts=None, complete=True):
    rates = {t: round(2.0 * dims[0] * dims[1] * dims[2] / min(ts) / 1e6, 1)
             for t, ts in samples.items()}
    checks = [{"stage": "check_shape", "tile": TILES[0], "rel_err": 0.0,
               "tolerance": 1e-5, "ok": True}]
    checks += [{"stage": "rate_shape", "tile": t, "rel_err": 0.0,
                "tolerance": 1e-5, "ok": True} for t in samples]
    gops_med = round(statistics.median(
        [2.0 * dims[0] * dims[1] * dims[2] / (statistics.median(ts) / 1e3) / 1e9
         for ts in samples.values()]), 1)
    return {"status": "ok", "relaxed_precision": relax, "packing": packing,
            "packing_attempts": attempts or [], "checks": checks,
            "rates": rates, "samples": samples,
            "gops_best": max(rates.values()), "gops_med": gops_med}


def sweep_receipt(device_name="Apple M9", architecture="applegpu_g99",
                  memory_size=32 * 2**30, mlx="0.32.3", version=3,
                  complete=True, f16=True, lanes_requested=None):
    rec = {
        "probe": "apple-matrix-capabilities/lane_sweep",
        "probe_sha256": "0" * 64,
        "mlx": mlx,
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "protocol": {"shape": list(SHAPE), "check_shape": [64, 1024, 256],
                     "tiles": list(TILES), "warmups": 2, "reps": 5,
                     "sample": "one call inside the timer", "mode": "multiply",
                     "seed": 1,
                     "lanes_requested": lanes_requested or
                     (["f16*f16->f32", "e4m3*e4m3->f32"] if f16 else ["e4m3*e4m3->f32"])},
        "results": {"ok": 2 if f16 else 1, "wrong": 0, "failed": 0,
                    "lanes_total": 2 if f16 else 1},
        "e4m3*e4m3->f32": lane_from_samples({TILES[0]: samples_for(2000.0),
                                             TILES[1]: samples_for(1980.0)}),
    }
    if f16:
        rec["f16*f16->f32"] = lane_from_samples({TILES[0]: samples_for(1000.0),
                                                 TILES[1]: samples_for(990.0)})
    if version is not None:
        rec["receipt_schema"] = version
    if complete is not None:
        rec["complete"] = complete
    return rec


def legacy_receipt(device_name="Apple M9", architecture="applegpu_g99",
                   memory_size=32 * 2**30, mlx="0.32.3"):
    """Pre-schema-2 shape: no receipt_schema, no protocol, no completion."""
    return {
        "probe": "apple-matrix-capabilities/lane_sweep",
        "mlx": mlx,
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "f16*f16->f32": {"status": "ok", "gops_best": 1000.0,
                         "rates": {"64x128x4": 1000.0}},
        "e4m3*e4m3->f32": {"status": "ok", "gops_best": 2000.0,
                           "rates": {"64x128x4": 2000.0}},
    }


def int_row(lane, tile, shape, rate, status="ok", stage="timed", reps=7):
    m, k, n = shape
    row = {"lane": lane, "tile": list(tile), "M": m, "K": k, "N": n,
           "status": status, "stage": stage, "packing": None,
           "packing_attempts": [],
           "checks": [{"stage": "check_shape", "rel_err": 0.0,
                       "tolerance": "exact", "ok": True},
                      {"stage": f"rate_shape_{m}x{k}x{n}", "rel_err": 0.0,
                       "tolerance": "exact", "ok": True}]}
    if status == "ok":
        mmin = 2.0 * m * k * n / (rate * 1e6)
        ts = [round(mmin * (1 + 0.01 * i), 3) for i in range(reps)]
        mb = min(ts) * (1 + 1e-9)
        row.update({"ms_best": mb, "ms_med": statistics.median(ts),
                    "ms_samples": ts, "gops_best": 2.0 * m * k * n / (mb * 1e6),
                    "gops_med": 2.0 * m * k * n / (statistics.median(ts) * 1e6)})
    return row


def integer_receipt(device_name="Apple M9", architecture="applegpu_g99",
                    memory_size=32 * 2**30, mlx="0.32.3", version=3,
                    lanes=("f16*f16->f32", "i8*i8->i32"), reps=7):
    shapes = [(2048, 4096, 4096), (4096, 4096, 4096)]
    rows = []
    for lane in lanes:
        rate = 1000.0 if lane.startswith("f16") else 2000.0
        for tile in ([64, 128, 4], [64, 64, 2]):
            for shape in shapes:
                rows.append(int_row(lane, tile, shape, rate * shape[0] / 2048.0,
                                    reps=reps))
    rec = {
        "probe": "apple-matrix-capabilities/integer_lanes",
        "probe_sha256": "0" * 64,
        "mlx": mlx,
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "relaxed": False, "calls": 4, "reps": reps,
        "protocol": {"shapes": [f"{m}x{k}x{n}" for m, k, n in shapes],
                     "check_shape": "128x256x256",
                     "tiles": ["64x128x4", "64x64x2"],
                     "lanes_requested": list(lanes),
                     "warmups": 2, "seed": 1, "reps": reps,
                     "sample": "batched timer", "packing_checked": "lo-first"},
        "complete": True,
        "rows": rows,
    }
    rec["results"] = {"rows_ok": sum(1 for r in rows if r["status"] == "ok"),
                      "rows_wrong": 0, "rows_failed": 0, "rows_total": len(rows)}
    if version is not None:
        rec["receipt_schema"] = version
    return rec


def geometry_receipt(device_name="Apple M9", architecture="applegpu_g99",
                     memory_size=32 * 2**30, version=3):
    shapes = {"shape_a": list(SHAPE)}
    cell = lane_from_samples({TILES[0]: samples_for(3000.0),
                              TILES[1]: samples_for(2900.0)})
    return {
        "probe": "apple-matrix-capabilities/geometry_sweep",
        "receipt_schema": version,
        "probe_sha256": "0" * 64,
        "mlx": "0.32.3",
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "protocol": {"shapes": shapes, "lanes_requested": ["f16*f16->f32"],
                     "tiles": list(TILES), "check_shape": [64, 1024, 256],
                     "rate_shape": list(SHAPE), "warmups": 2, "reps": 5,
                     "seed": 1, "mode": "multiply", "sample": "timer"},
        "lane_meta": {"f16*f16->f32": {"relaxed_precision": False,
                                       "packing": None, "packing_attempts": []}},
        "complete": True,
        "lanes": {"f16*f16->f32": {"shape_a": cell}},
        "results": {"cells_ok": 1, "cells_total": 1},
    }


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "matrix"))
        os.makedirs(os.path.join(self.root, "receipts", "m9"))
        for rel in ("matrix/schema.json", "receipts/schema.json"):
            shutil.copy(os.path.join(ROOT, *rel.split("/")),
                        os.path.join(self.root, *rel.split("/")))
        self.write_receipt(sweep_receipt())

    def tearDown(self):
        self.tmp.cleanup()

    def write_receipt(self, rec, name="lane_sweep.json"):
        path = os.path.join(self.root, "receipts", "m9", name)
        with open(path, "w") as f:
            json.dump(rec, f)
        return f"receipts/m9/{name}"

    def rewrite_raw(self, text, name="lane_sweep.json"):
        path = os.path.join(self.root, "receipts", "m9", name)
        with open(path, "w") as f:
            f.write(text)

    def cell(self, rate=2000.0, ratio=2.0, receipt="receipts/m9/lane_sweep.json"):
        return {"gops_best": rate, "vs_f16": ratio, "receipt": receipt}

    def cap(self, cells, **chip_overrides):
        chip = {"device_name": "Apple M9", "architecture": "applegpu_g99",
                "gpu_cores": 10, "memory_gb": 32, "evidence": "observed",
                "mlx_at_measurement": "0.32.3", "float_lanes": cells}
        chip.update(chip_overrides)
        return {"schema": 1, "chips": {"m9": chip}}

    def assert_error(self, errors, needle):
        self.assertTrue(errors, "expected an error, got none")
        self.assertTrue(any(needle in e for e in errors),
                        f"no error containing {needle!r} in {errors!r}")


class TestValidator(Fixture):
    def test_valid_fixture_passes(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assertEqual(bm.validate(cap, self.root), [])

    def test_v2_pre_flag_receipt_accepted_when_coverage_infers(self):
        # Schema-2 receipts predate explicit requested-work fields; coverage is
        # inferred from surviving rows and labelled as such in the views.
        rec = sweep_receipt(version=2, complete=None)
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assertEqual(bm.validate(cap, self.root), [])

    def test_incomplete_checkpoint_never_citable(self):
        self.write_receipt(sweep_receipt(complete=False))
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "incomplete checkpoint")

    def test_unproven_coverage_rejected(self):
        rec = sweep_receipt()
        del rec["e4m3*e4m3->f32"]  # requested but never recorded
        rec["results"] = {"ok": 1, "wrong": 0, "failed": 0, "lanes_total": 2}
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "completion unproven")

    def test_legacy_receipt_not_citable(self):
        name = self.write_receipt(legacy_receipt(), "legacy.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell(receipt=name)})
        self.assert_error(bm.validate(cap, self.root), "predates schema 2")

    def test_unsupported_receipt_version_rejected(self):
        rec = sweep_receipt()
        rec["receipt_schema"] = 999
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        errors = bm.validate(cap, self.root)
        self.assert_error(errors, "schema:")
        self.assert_error(errors, "unsupported receipt_schema")

    # --- finding 3: numerical consistency, each mutation must fail ----------

    def test_passing_check_with_large_error_rejected(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["checks"][1]["rel_err"] = 1.0  # tolerance 1e-5
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "exceeds tolerance")

    def test_missing_rate_shape_check_rejected(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["checks"] = [
            c for c in rec["e4m3*e4m3->f32"]["checks"]
            if c.get("stage") != "rate_shape" or c.get("tile") != TILES[1]]
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "no passing rate_shape check")

    def test_samples_scaled_without_rates_rejected(self):
        rec = sweep_receipt()
        lane = rec["e4m3*e4m3->f32"]
        lane["samples"][TILES[0]] = [t * 100 for t in lane["samples"][TILES[0]]]
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "samples-derived")

    def test_missing_samples_rejected(self):
        rec = sweep_receipt()
        del rec["e4m3*e4m3->f32"]["samples"]
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "without timing samples")

    def test_wrong_sample_count_rejected(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["samples"][TILES[0]] = \
            rec["e4m3*e4m3->f32"]["samples"][TILES[0]][:3]
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "sample count")

    def test_baseline_gops_doubled_rejected(self):
        rec = sweep_receipt()
        rec["f16*f16->f32"]["gops_best"] *= 2
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "gops_best")

    def test_baseline_rates_emptied_rejected(self):
        rec = sweep_receipt()
        rec["f16*f16->f32"]["rates"] = {}
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "without tile rates")

    def test_uncited_geometry_cell_tampered_rejected(self):
        rec = geometry_receipt()
        rec["lanes"]["f16*f16->f32"]["shape_a"]["gops_best"] = 1e10
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "geometry_sweep.json")

    def test_non_finite_number_anywhere_rejected(self):
        path = os.path.join(self.root, "receipts", "m9", "geometry_sweep.json")
        with open(path, "w") as f:
            json.dump(geometry_receipt(), f)
        with open(path) as f:
            text = f.read()
        self.assertIn('"memory_size": 34359738368', text)
        self.rewrite_raw(text.replace('"memory_size": 34359738368',
                                      '"memory_size": 1e999'),
                         "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "non-finite number")

    def test_v3_integer_missing_requested_rows_rejected(self):
        rec = integer_receipt()
        rec["rows"] = [r for r in rec["rows"] if r["M"] != 4096]
        rec["results"] = {"rows_ok": len(rec["rows"]), "rows_wrong": 0,
                          "rows_failed": 0, "rows_total": len(rec["rows"])}
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "no terminal row")

    def test_v3_integer_extra_rows_rejected(self):
        rec = integer_receipt()
        rec["rows"].append(int_row("i8*i8->i32", [64, 32, 4], SHAPE, 2000.0))
        rec["results"]["rows_total"] = len(rec["rows"])
        rec["results"]["rows_ok"] = len(rec["rows"])
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "outside the requested work")

    def test_integer_ms_best_mismatch_rejected(self):
        rec = integer_receipt()
        row = next(r for r in rec["rows"] if r["lane"] == "i8*i8->i32"
                   and r["M"] == 2048 and r["tile"] == [64, 128, 4])
        row["ms_best"] *= 2
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "ms_best")

    def test_removed_float_tile_outcome_rejected(self):
        rec = sweep_receipt()
        lane = rec["e4m3*e4m3->f32"]
        lane["rates"].pop(TILES[1])
        lane["samples"].pop(TILES[1])
        lane["checks"] = [c for c in lane["checks"] if c.get("tile") != TILES[1]]
        lane["gops_best"] = max(lane["rates"].values())
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell(rate=lane["gops_best"],
                                                    ratio=round(lane["gops_best"] / rec["f16*f16->f32"]["gops_best"], 3))})
        self.assert_error(bm.validate(cap, self.root), "no measurement or failure outcome")

    def test_duplicate_integer_terminal_row_rejected(self):
        rec = integer_receipt()
        rec["rows"].append(dict(rec["rows"][0]))
        rec["results"]["rows_total"] += 1
        rec["results"]["rows_ok"] += 1
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "duplicate terminal")

    def test_empty_integer_protocol_tiles_rejected(self):
        rec = integer_receipt()
        rec["protocol"]["tiles"] = []
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "nonempty parseable")

    def test_geometry_empty_shapes_map_rejected(self):
        rec = geometry_receipt()
        rec["protocol"]["shapes"] = {}
        rec["lanes"]["f16*f16->f32"]["shape_a"]["gops_best"] = 1e10
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        errors = bm.validate(cap, self.root)
        self.assert_error(errors, "nonempty protocol.shapes")
        self.assert_error(errors, "no valid dimensions")

    def test_geometry_label_with_dims_removed_rejected(self):
        rec = geometry_receipt()
        del rec["protocol"]["shapes"]["shape_a"]
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "no valid dimensions")

    def test_integer_check_stage_dims_mismatch_rejected(self):
        rec = integer_receipt()
        row = next(r for r in rec["rows"] if r["lane"] == "i8*i8->i32")
        for c in row["checks"]:
            if c["stage"].startswith("rate_shape"):
                c["stage"] = "rate_shape_1x1x1"
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "!= row")

    def test_integer_reps_conflict_rejected(self):
        rec = integer_receipt()
        rec["protocol"]["reps"] = 99
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "protocol.reps 99 != receipt reps 7")

    def test_integer_zero_ms_best_is_an_error_not_a_traceback(self):
        rec = integer_receipt()
        row = next(r for r in rec["rows"] if r["lane"] == "i8*i8->i32")
        row["ms_best"] = 0
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        errors = bm.validate(cap, self.root)
        self.assert_error(errors, "positive finite")

    def test_incomplete_uncited_checkpoint_allowed_but_not_citable(self):
        self.write_receipt(sweep_receipt(complete=False), "checkpoint.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assertEqual(bm.validate(cap, self.root), [])
        md, _ = bm.build(cap, self.root)
        self.assertIn("incomplete (not citable)", md)
        cap2 = self.cap({"e4m3*e4m3->f32": self.cell(
            receipt="receipts/m9/checkpoint.json")})
        self.assert_error(bm.validate(cap2, self.root), "incomplete checkpoint")

    def test_packed_selected_order_needs_correctness_check(self):
        rec = sweep_receipt(lanes_requested=["fp4*fp4->f32"])
        del rec["f16*f16->f32"]
        del rec["e4m3*e4m3->f32"]
        lane = lane_from_samples({TILES[0]: samples_for(500.0)},
                                 attempts=["lo", "hi"])
        lane["packing"] = "hi"
        for c in lane["checks"]:
            c["order"] = "lo"
            c["exploratory"] = True
        rec["fp4*fp4->f32"] = lane
        rec["results"] = {"ok": 1, "wrong": 0, "failed": 0, "lanes_total": 1}
        self.write_receipt(rec)
        cap = self.cap({"fp4*fp4->f32": self.cell(rate=500.0, ratio=0.5)})
        self.assert_error(bm.validate(cap, self.root), "no passing correctness check")

    # --- fourth review: geometry tiles, exploratory, reps, malformed input ---

    def test_geometry_v3_missing_tiles_rejected(self):
        # Schema (if/then) enforces the v3 tile request before any semantics.
        rec = geometry_receipt()
        del rec["protocol"]["tiles"]
        errors = bm.receipt_schema_errors("geometry_sweep.json", rec, self.root)
        self.assertTrue(errors, "v3 geometry without protocol.tiles must fail")
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "schema:")

    def test_geometry_v3_empty_tiles_rejected(self):
        rec = geometry_receipt()
        rec["protocol"]["tiles"] = []
        errors = bm.receipt_schema_errors("geometry_sweep.json", rec, self.root)
        self.assertTrue(errors, "v3 geometry with empty protocol.tiles must fail")
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "schema:")

    def test_geometry_removed_tile_outcome_rejected(self):
        rec = geometry_receipt()
        cell = rec["lanes"]["f16*f16->f32"]["shape_a"]
        cell["rates"].pop(TILES[1])
        cell["samples"].pop(TILES[1])
        cell["checks"] = [c for c in cell["checks"] if c.get("tile") != TILES[1]]
        cell["gops_best"] = max(cell["rates"].values())
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "no measurement or failure outcome")

    def test_exploratory_failed_check_on_nonpacked_lane_rejected(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["checks"][0].update(
            {"ok": False, "rel_err": 1.0, "exploratory": True})
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "exploratory outside packing")

    def test_exploratory_timing_check_rejected(self):
        rec = sweep_receipt()
        lane = rec["e4m3*e4m3->f32"]
        timing = next(c for c in lane["checks"] if c.get("stage") == "rate_shape")
        timing["exploratory"] = True
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "exploratory outside packing")

    def test_exploratory_selected_order_check_rejected(self):
        rec = sweep_receipt(lanes_requested=["fp4*fp4->f32"])
        del rec["f16*f16->f32"]
        del rec["e4m3*e4m3->f32"]
        lane = lane_from_samples({TILES[0]: samples_for(500.0)},
                                 attempts=["lo", "hi"])
        lane["packing"] = "lo"
        lane["checks"][0]["order"] = "lo"
        lane["checks"][0]["exploratory"] = True
        rec["fp4*fp4->f32"] = lane
        rec["results"] = {"ok": 1, "wrong": 0, "failed": 0, "lanes_total": 1}
        self.write_receipt(rec)
        cap = self.cap({"fp4*fp4->f32": self.cell(rate=500.0, ratio=0.5)})
        self.assert_error(bm.validate(cap, self.root), "no passing correctness check")

    def _packed_lane(self, packing, attempts, extra_check=None):
        lane = lane_from_samples({TILES[0]: samples_for(500.0),
                                  TILES[1]: samples_for(490.0)},
                                 attempts=attempts)
        lane["packing"] = packing
        lane["checks"][0]["order"] = packing             # passing selected-order
        lane["checks"][0]["exploratory"] = False
        if extra_check:
            lane["checks"].append(extra_check)
        return lane

    def _packed_receipt(self, lane):
        rec = sweep_receipt(
            lanes_requested=["f16*f16->f32", "e4m3*e4m3->f32", "fp4*fp4->f32"])
        rec["fp4*fp4->f32"] = lane
        status = lane["status"]
        rec["results"] = {"ok": 3 if status == "ok" else 2,
                          "wrong": 1 if status == "wrong" else 0,
                          "failed": 1 if status == "failed" else 0,
                          "lanes_total": 3}
        return rec

    def test_exploratory_selected_order_duplicate_rejected(self):
        # Additive regression: the selected-order path already passed, but a
        # second exploratory entry for the same order must not be exempt.
        lane = self._packed_lane(
            "lo", ["lo", "hi"],
            extra_check={"stage": "check_shape", "tile": TILES[0], "order": "lo",
                         "exploratory": True, "ok": False, "rel_err": 1.0,
                         "tolerance": 1e-5})
        self.write_receipt(self._packed_receipt(lane))
        cap = self.cap({"fp4*fp4->f32": self.cell(rate=500.0, ratio=0.5)})
        self.assert_error(bm.validate(cap, self.root),
                          "marks the selected order 'lo' exploratory")

    def test_exploratory_order_not_attempted_rejected(self):
        lane = self._packed_lane(
            "lo", ["lo"],
            extra_check={"stage": "check_shape", "tile": TILES[0], "order": "hi",
                         "exploratory": True, "ok": False, "rel_err": 1.0,
                         "tolerance": 1e-5})
        self.write_receipt(self._packed_receipt(lane))
        cap = self.cap({"fp4*fp4->f32": self.cell(rate=500.0, ratio=0.5)})
        self.assert_error(bm.validate(cap, self.root), "not in packing_attempts")

    def test_valid_hi_fallback_still_passes(self):
        lane = self._packed_lane("hi", ["lo", "hi"])
        lane["checks"][0].update({"order": "hi", "exploratory": False})
        lane["checks"].insert(0, {"stage": "check_shape", "tile": TILES[0],
                                  "order": "lo", "exploratory": True, "ok": False,
                                  "rel_err": 1.0, "tolerance": 1e-5})
        self.write_receipt(self._packed_receipt(lane))
        cap = self.cap({"fp4*fp4->f32": self.cell(rate=500.0, ratio=0.5)})
        self.assertEqual(bm.validate(cap, self.root), [])

    def test_all_candidates_failed_without_packing_remains_valid(self):
        lane = self._packed_lane("lo", ["lo", "hi"])
        lane["status"] = "wrong"
        lane["packing"] = None
        lane["checks"] = [
            {"stage": "check_shape", "tile": TILES[0], "order": "lo",
             "exploratory": True, "ok": False, "rel_err": 1.0, "tolerance": 1e-5},
            {"stage": "check_shape", "tile": TILES[0], "order": "hi",
             "exploratory": True, "ok": False, "rel_err": 1.0, "tolerance": 1e-5},
        ]
        for key in ("rates", "samples", "gops_best", "gops_med"):
            lane.pop(key, None)
        self.write_receipt(self._packed_receipt(lane))
        # The failed lane is archive evidence; the cited cell uses a good lane.
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assertEqual(bm.validate(cap, self.root), [])

    def test_integer_v3_missing_receipt_reps_rejected(self):
        rec = integer_receipt()
        del rec["reps"]
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "reps")

    def test_integer_v3_protocol_reps_string_rejected(self):
        rec = integer_receipt()
        rec["protocol"]["reps"] = "seven"
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "reps")

    def test_integer_v3_zero_or_boolean_reps_rejected(self):
        for bad in (0, True):
            rec = integer_receipt()
            rec["reps"] = bad
            rec["protocol"]["reps"] = bad
            self.write_receipt(rec, "integer_lanes.json")
            cap = self.cap({"i8*i8->i32": self.cell(
                rate=2000.0, ratio=2.0, receipt="receipts/m9/integer_lanes.json")})
            self.assert_error(bm.validate(cap, self.root), "reps")

    def test_integer_sample_count_mismatch_rejected(self):
        rec = integer_receipt()
        row = next(r for r in rec["rows"] if r["lane"] == "i8*i8->i32")
        row["ms_samples"] = row["ms_samples"][:3]
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt="receipts/m9/integer_lanes.json")})
        self.assert_error(bm.validate(cap, self.root), "sample count")

    def test_malformed_integer_tile_request_errors_without_traceback(self):
        for bad in (["not-a-tile"], ["0x128x4"], ["64x128"], ["64x128x3"]):
            rec = integer_receipt()
            rec["protocol"]["tiles"] = bad
            self.write_receipt(rec, "integer_lanes.json")
            cap = self.cap({"i8*i8->i32": self.cell(
                rate=2000.0, ratio=2.0, receipt="receipts/m9/integer_lanes.json")})
            errors = bm.validate(cap, self.root)  # must not raise
            self.assertTrue(errors)

    def test_malformed_geometry_tile_request_errors_without_traceback(self):
        rec = geometry_receipt()
        rec["protocol"]["tiles"] = ["0x128x4"]
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        errors = bm.validate(cap, self.root)  # modulo-by-zero guard
        self.assert_error(errors, "non-positive")

    def test_malformed_integer_shape_request_errors_without_traceback(self):
        rec = integer_receipt()
        rec["protocol"]["shapes"] = ["not-a-shape"]
        rec["protocol"]["tiles"] = ["64x128x4"]
        rec["rows"] = [r for r in rec["rows"] if r["tile"] == [64, 128, 4]]
        rec["results"] = {"rows_ok": len(rec["rows"]), "rows_wrong": 0,
                          "rows_failed": 0, "rows_total": len(rec["rows"])}
        self.write_receipt(rec, "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(
            rate=2000.0, ratio=2.0, receipt="receipts/m9/integer_lanes.json")})
        errors = bm.validate(cap, self.root)  # must not raise
        self.assertTrue(errors)

    def test_v3_packed_lane_without_attempts_rejected(self):
        rec = sweep_receipt(lanes_requested=["fp4*fp4->f32"])
        rec["fp4*fp4->f32"] = lane_from_samples({TILES[0]: samples_for(500.0)},
                                                attempts=None)
        rec["fp4*fp4->f32"]["packing"] = "lo"
        del rec["fp4*fp4->f32"]["packing_attempts"]
        rec["results"] = {"ok": 1, "wrong": 0, "failed": 0, "lanes_total": 1}
        del rec["f16*f16->f32"]
        del rec["e4m3*e4m3->f32"]
        self.write_receipt(rec)
        cap = self.cap({"fp4*fp4->f32": self.cell(rate=500.0, ratio=0.5)})
        self.assert_error(bm.validate(cap, self.root), "packing_attempts")

    def test_ok_status_with_failed_check_rejected_by_walk(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["checks"][1]["ok"] = False
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "recorded check failed")

    def test_deleted_checks_under_ok_rejected_by_walk(self):
        rec = sweep_receipt()
        del rec["e4m3*e4m3->f32"]["checks"]
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "without any recorded check")

    def test_protocol_mode_and_shape_enforced(self):
        rec = sweep_receipt()
        rec["protocol"]["mode"] = "add"
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "schema:")
        rec = sweep_receipt()
        rec["protocol"]["shape"] = [1024, 4096, 4096]
        self.write_receipt(rec)
        self.assert_error(bm.validate(cap, self.root), "protocol.shape")

    # --- structural / identity / fidelity (carried from the first pass) -----

    def test_fabricated_lane_rejected(self):
        cap = self.cap({"made*up->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "lane not in")

    def test_tampered_rate_and_ratio_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell(rate=4000.0, ratio=4.0)})
        self.assert_error(bm.validate(cap, self.root), "!= receipt's")

    def test_baseline_status_must_be_ok(self):
        rec = sweep_receipt()
        rec["f16*f16->f32"]["status"] = "wrong"
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "baseline")

    def test_cited_lane_status_must_be_ok(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["status"] = "wrong"
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "status is 'wrong'")

    def test_receipt_path_escape_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell(receipt="../secret.json")})
        self.assert_error(bm.validate(cap, self.root), "receipt must be receipts/m9")

    def test_receipt_from_other_chip_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell(receipt="receipts/m8/other.json")})
        self.assert_error(bm.validate(cap, self.root), "receipt must be receipts/m9")

    def test_symlinked_receipt_escape_rejected(self):
        outside = os.path.join(self.root, "outside.json")
        with open(outside, "w") as f:
            json.dump(sweep_receipt(), f)
        link = os.path.join(self.root, "receipts", "m9", "esc.json")
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        cap = self.cap({"e4m3*e4m3->f32": self.cell(receipt="receipts/m9/esc.json")})
        self.assert_error(bm.validate(cap, self.root), "escapes")

    def test_device_mismatch_and_bad_cores_rejected(self):
        rec = sweep_receipt(device_name="Apple M8", memory_size=64 * 2**30)
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()}, gpu_cores=0)
        errors = bm.validate(cap, self.root)
        self.assert_error(errors, "device_name")
        self.assert_error(errors, "memory_size")
        self.assert_error(errors, "gpu_cores")

    def test_receipt_gpu_cores_mismatch_rejected_when_present(self):
        rec = sweep_receipt()
        rec["device"]["gpu_cores"] = 12
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "receipt gpu_cores 12")

    def test_missing_memory_size_on_cited_receipt_rejected(self):
        rec = sweep_receipt()
        del rec["device"]["memory_size"]
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "no device.memory_size")

    def test_mlx_mismatch_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()}, mlx_at_measurement="0.31.0")
        self.assert_error(bm.validate(cap, self.root), "mlx")

    def test_nonfinite_cell_rate_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell(rate=float("inf"), ratio=2.0)})
        self.assert_error(bm.validate(cap, self.root), "finite and positive")

    def test_nan_receipt_text_rejected(self):
        self.rewrite_raw(
            '{"probe": "x", "mlx": "0.32.3", "device": {"device_name": '
            '"Apple M9", "architecture": "applegpu_g99"}, "f16*f16->f32": '
            '{"status": "ok", "gops_best": NaN}}')
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "non-strict JSON")

    def test_uncited_malformed_receipt_fails_walk(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["status"] = "bogus"
        self.write_receipt(rec, "bad.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "bad.json")

    def test_reported_chip_with_cells_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()}, evidence="reported")
        self.assert_error(bm.validate(cap, self.root), "cannot carry measured lanes")

    def test_integer_receipt_rate_mismatch_rejected(self):
        name = self.write_receipt(integer_receipt(), "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=3000.0, ratio=3.0, receipt=name)})
        self.assert_error(bm.validate(cap, self.root), "!= receipt's")

    def test_integer_receipt_valid_match_passes(self):
        name = self.write_receipt(integer_receipt(), "integer_lanes.json")
        cap = self.cap({"i8*i8->i32": self.cell(rate=2000.0, ratio=2.0, receipt=name)})
        self.assertEqual(bm.validate(cap, self.root), [])

    def test_unknown_chips_overlap_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        cap["unknown_chips"] = ["m9", "m1"]
        self.assert_error(bm.validate(cap, self.root), "unknown_chips contains measured chip")

    def test_schema_violation_reported(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()}, gpu_cores="ten")
        self.assert_error(bm.validate(cap, self.root), "schema:")


class TestViews(Fixture):
    def test_views_flag_wide_sample_spread(self):
        rec = sweep_receipt()
        ts = rec["e4m3*e4m3->f32"]["samples"][TILES[0]]
        ts[-1] = round(ts[-1] * 3, 3)  # spread > 2x, minimum (and rate) unchanged
        self.write_receipt(rec)
        cell = rec["e4m3*e4m3->f32"]
        cap = self.cap({"e4m3*e4m3->f32": self.cell(
            rate=cell["gops_best"],
            ratio=round(cell["gops_best"] / rec["f16*f16->f32"]["gops_best"], 3))})
        self.assertEqual(bm.validate(cap, self.root), [])
        md, csv_text = bm.build(cap, self.root)
        self.assertIn("⚠", md)
        self.assertIn("spread>2x", csv_text)

    def test_uncited_geometry_receipt_spread_rendered(self):
        rec = geometry_receipt()
        ts = rec["lanes"]["f16*f16->f32"]["shape_a"]["samples"][TILES[0]]
        ts[-1] = round(ts[-1] * 3, 3)  # spread > 2x, minimum (and rate) unchanged
        self.write_receipt(rec, "geometry_sweep.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assertEqual(bm.validate(cap, self.root), [])
        md, _ = bm.build(cap, self.root)
        self.assertIn("geometry_sweep.json` | `apple-matrix-capabilities/geometry_sweep`", md)
        self.assertIn("× ⚠", md)
        self.assertRegex(md, r"geometry_sweep\.json.*shape_a")

    def test_views_have_per_receipt_baselines(self):
        name = self.write_receipt(integer_receipt(), "integer_lanes.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell(),
                        "i8*i8->i32": self.cell(rate=2000.0, ratio=2.0,
                                                receipt=name)})
        md, csv_text = bm.build(cap, self.root)
        self.assertIn("f16 baselines (GOPS, by receipt)", md)
        self.assertIn("baseline_gops", csv_text)
        self.assertIn("baseline_receipt", csv_text)
        self.assertNotIn("f16 denominator per chip", md)

    def test_completion_labels_reported_in_index(self):
        name = self.write_receipt(sweep_receipt(version=2, complete=None),
                                  "lane_sweep_v2.json")
        self.write_receipt(sweep_receipt(version=3, complete=None),
                           "lane_sweep_v3_coverage.json")
        cap = self.cap({"e4m3*e4m3->f32": self.cell(receipt=name)})
        md, _ = bm.build(cap, self.root)
        self.assertIn("coverage (inferred, v2)", md)
        self.assertIn("coverage (v3)", md)


if __name__ == "__main__":
    unittest.main()
