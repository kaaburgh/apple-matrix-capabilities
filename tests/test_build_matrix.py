#!/usr/bin/env python3
"""Regression tests for the receipt-backed validation in tools/build_matrix.py.

Run from the repository root:

    python -m unittest discover -s tests -v

They exercise the validator against synthetic schema-2 fixtures in a temporary
root, so no hardware and no real receipts are needed. jsonschema must be
installed.
"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "build_matrix", os.path.join(ROOT, "tools", "build_matrix.py"))
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

PROTO = {"shape": [2048, 4096, 4096], "check_shape": [64, 1024, 256],
         "tiles": ["64x128x4", "64x64x2"], "warmups": 2, "reps": 3,
         "sample": "timer test", "mode": "multiply", "seed": 1,
         "lanes_requested": ["f16*f16->f32", "e4m3*e4m3->f32"]}


def sweep_lane(rate, ok=True, samples=None):
    return {
        "status": "ok" if ok else "wrong",
        "relaxed_precision": False,
        "packing": None,
        "checks": [{"stage": "rate_shape", "tile": "64x128x4", "rel_err": 0.0,
                    "tolerance": 1e-5, "ok": ok}],
        "rates": {"64x128x4": rate, "64x64x2": rate * 0.99},
        "samples": samples or {"64x128x4": [1.0, 1.01], "64x64x2": [1.02, 1.03]},
        "gops_best": rate,
    }


def sweep_receipt(device_name="Apple M9", architecture="applegpu_g99",
                  memory_size=32 * 2**30, mlx="0.32.3", complete=True):
    rec = {
        "probe": "apple-matrix-capabilities/lane_sweep",
        "receipt_schema": 2,
        "probe_sha256": "0" * 64,
        "mlx": mlx,
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "protocol": json.loads(json.dumps(PROTO)),
        "results": {"ok": 2, "wrong": 0, "failed": 0, "lanes_total": 2},
        "f16*f16->f32": sweep_lane(1000.0),
        "e4m3*e4m3->f32": sweep_lane(2000.0),
    }
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


def integer_receipt(device_name="Apple M9", architecture="applegpu_g99",
                    memory_size=32 * 2**30, mlx="0.32.3"):
    rows = []
    for lane, rate in (("f16*f16->f32", 1000.0), ("i8*i8->i32", 2000.0)):
        for tile in ([64, 128, 4], [64, 64, 2]):
            for shape in ([2048, 4096, 4096], [4096, 4096, 4096]):
                rows.append({"lane": lane, "tile": tile, "status": "ok",
                             "M": shape[0], "K": shape[1], "N": shape[2],
                             "ms_samples": [10.0, 10.1],
                             "gops_best": rate if tile == [64, 128, 4] else rate * 0.99,
                             "gops_med": rate * 0.99,
                             "checks": [{"stage": "check_shape", "rel_err": 0.0,
                                         "tolerance": "exact", "ok": True}]})
    return {
        "probe": "apple-matrix-capabilities/integer_lanes",
        "receipt_schema": 2,
        "probe_sha256": "0" * 64,
        "mlx": mlx,
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "protocol": {"shapes": ["2048x4096x4096", "4096x4096x4096"],
                     "check_shape": "128x256x256",
                     "tiles": ["64x128x4", "64x64x2"],
                     "warmups": 2, "seed": 1, "sample": "timer test"},
        "complete": True,
        "results": {"rows_ok": 8, "rows_wrong": 0, "rows_failed": 0,
                    "rows_total": 8},
        "rows": rows,
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

    def test_pre_flag_receipt_accepted_when_coverage_proves_completion(self):
        # Receipts written before the `complete` flag exist; coverage
        # (lanes recorded == lanes requested, consistent counts) proves them.
        self.write_receipt(sweep_receipt(complete=None))
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

    def test_fabricated_lane_rejected(self):
        cap = self.cap({"made*up->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "lane not in")

    def test_tampered_rate_and_ratio_rejected(self):
        # Doubling the rate and the ratio keeps them consistent with each other,
        # but not with the receipt.
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

    def test_mlx_mismatch_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()}, mlx_at_measurement="0.31.0")
        self.assert_error(bm.validate(cap, self.root), "mlx")

    def test_nonfinite_rejected(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell(rate=float("inf"), ratio=2.0)})
        errors = bm.validate(cap, self.root)
        self.assert_error(errors, "finite and positive")

    def test_nan_receipt_text_rejected(self):
        path = os.path.join(self.root, "receipts", "m9", "nan.json")
        with open(path, "w") as f:
            f.write('{"probe": "x", "mlx": "0.32.3", "device": {"device_name": '
                    '"Apple M9", "architecture": "applegpu_g99"}, "f16*f16->f32": '
                    '{"status": "ok", "gops_best": NaN}}')
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "non-strict JSON")

    def test_nonfinite_number_in_receipt_rejected_by_walk(self):
        path = os.path.join(self.root, "receipts", "m9", "lane_sweep.json")
        with open(path) as f:
            text = f.read()
        self.assertIn('"64x128x4": 2000.0', text)
        with open(path, "w") as f:
            f.write(text.replace('"64x128x4": 2000.0', '"64x128x4": 1e999', 1))
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "not finite")

    def test_ok_status_with_failed_check_rejected_by_walk(self):
        rec = sweep_receipt()
        rec["e4m3*e4m3->f32"]["checks"][0]["ok"] = False
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "recorded check failed")

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
        rec["e4m3*e4m3->f32"]["samples"] = {"64x128x4": [1.0, 3.0],
                                            "64x64x2": [1.0, 1.0]}
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assertEqual(bm.validate(cap, self.root), [])
        md, csv_text = bm.build(cap, self.root)
        self.assertIn("⚠", md)
        self.assertIn("spread>2x", csv_text)

    def test_views_have_baseline_and_receipt_index(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        md, _ = bm.build(cap, self.root)
        self.assertIn("f16 baseline (GOPS)", md)
        self.assertIn("f16 denominator per chip", md)
        self.assertIn("## Receipts", md)
        self.assertIn("receipts/m9/lane_sweep.json", md)

    def test_completion_label_reported_in_index(self):
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        md, _ = bm.build(cap, self.root)
        self.assertIn("| flag |", md)


if __name__ == "__main__":
    unittest.main()
