#!/usr/bin/env python3
"""Regression tests for the receipt-backed validation in tools/build_matrix.py.

Run from the repository root:

    python -m unittest discover -s tests -v

They exercise the validator against synthetic fixtures in a temporary root, so
no hardware and no real receipts are needed. jsonschema must be installed.
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


def mapping_receipt(device_name="Apple M9", architecture="applegpu_g99",
                    memory_size=32 * 2**30, mlx="0.32.3"):
    return {
        "probe": "test",
        "mlx": mlx,
        "device": {"device_name": device_name, "architecture": architecture,
                   "memory_size": memory_size},
        "f16*f16->f32": {"status": "ok", "gops_best": 1000.0,
                         "rates": {"64x128x4": 1000.0, "64x64x2": 990.0}},
        "e4m3*e4m3->f32": {"status": "ok", "gops_best": 2000.0,
                           "rates": {"64x128x4": 2000.0, "64x64x2": 1990.0}},
    }


def integer_receipt(device_name="Apple M9", architecture="applegpu_g99",
                    memory_size=32 * 2**30, mlx="0.32.3"):
    rows = []
    for lane, rate in (("f16*f16->f32", 1000.0), ("i8*i8->i32", 2000.0)):
        for tile in ([64, 128, 4], [64, 64, 2]):
            rows.append({"lane": lane, "tile": tile, "status": "ok", "rel_err": 0.0,
                         "M": 2048, "K": 4096, "N": 4096,
                         "gops_best": rate if tile == [64, 128, 4] else rate * 0.99,
                         "gops_med": rate * 0.99})
    return {"mlx": mlx, "device": {"device_name": device_name,
                                   "architecture": architecture,
                                   "memory_size": memory_size},
            "rows": rows}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "matrix"))
        os.makedirs(os.path.join(self.root, "receipts", "m9"))
        shutil.copy(os.path.join(ROOT, "matrix", "schema.json"),
                    os.path.join(self.root, "matrix", "schema.json"))
        self.write_receipt(mapping_receipt())

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

    def test_fabricated_lane_rejected(self):
        cap = self.cap({"made*up->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "lane not in")

    def test_tampered_rate_and_ratio_rejected(self):
        # Doubling the rate and the ratio keeps them consistent with each other,
        # but not with the receipt.
        cap = self.cap({"e4m3*e4m3->f32": self.cell(rate=4000.0, ratio=4.0)})
        self.assert_error(bm.validate(cap, self.root), "!= receipt's")

    def test_baseline_status_must_be_ok(self):
        rec = mapping_receipt()
        rec["f16*f16->f32"]["status"] = "wrong"
        self.write_receipt(rec)
        cap = self.cap({"e4m3*e4m3->f32": self.cell()})
        self.assert_error(bm.validate(cap, self.root), "baseline")

    def test_cited_lane_status_must_be_ok(self):
        rec = mapping_receipt()
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

    def test_device_mismatch_and_bad_cores_rejected(self):
        rec = mapping_receipt(device_name="Apple M8", memory_size=64 * 2**30)
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


if __name__ == "__main__":
    unittest.main()
