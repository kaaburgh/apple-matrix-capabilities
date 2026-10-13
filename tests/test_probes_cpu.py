#!/usr/bin/env python3
"""CPU-only tests for probes/_common.py, the probes' pure helpers, and the
probe-to-validator contract.

Run from the repository root:

    python -m unittest discover -s tests -v

The probes import `mlx.core` at module level (and call `mx.set_memory_limit`
during import), so a stub `mlx` module is installed first. The probe `main()`
paths are then driven with stubbed check/timing seams: every receipt a probe
writes is fed back through the validator's own receipt schema and semantic
rules, so writer and validator must agree on subsets, failures and non-finite
numerics. Nothing here touches a GPU.
"""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "probes")


def _install_mlx_stub():
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.__version__ = "test"
    core.device_info = lambda: {"device_name": "Apple Test",
                                "architecture": "applegpu_test",
                                "memory_size": 8 * 2**30,
                                "max_recommended_working_set_size": 1}
    core.set_memory_limit = lambda *a, **k: None
    core.set_cache_limit = lambda *a, **k: None
    for dtype in ("float16", "float32", "bfloat16", "int32"):
        setattr(core, dtype, dtype)
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_install_mlx_stub()
sys.path.insert(0, PROBES)
common = _load("_common", os.path.join(PROBES, "_common.py"))
lane_sweep = _load("lane_sweep", os.path.join(PROBES, "lane_sweep.py"))
integer_lanes = _load("integer_lanes", os.path.join(PROBES, "integer_lanes.py"))
geometry_sweep = _load("geometry_sweep", os.path.join(PROBES, "geometry_sweep.py"))

_bm_spec = importlib.util.spec_from_file_location(
    "build_matrix", os.path.join(ROOT, "tools", "build_matrix.py"))
bm = importlib.util.module_from_spec(_bm_spec)
sys.modules["build_matrix"] = bm
_bm_spec.loader.exec_module(bm)


def validate_probe_receipt(testcase, path):
    """Feed a probe-written receipt through both validator layers."""
    with open(path) as f:
        rec = json.load(f)
    rel = os.path.relpath(path, os.path.dirname(os.path.dirname(path)))
    testcase.assertEqual(bm.receipt_schema_errors(rel, rec, ROOT), [])
    testcase.assertEqual(bm.receipt_semantic_errors(rel, rec), [])
    return rec


def run_main(testcase, module, argv):
    old_argv = sys.argv
    sys.argv = [module.__file__] + argv
    try:
        return module.main()
    finally:
        sys.argv = old_argv


class TestCommon(unittest.TestCase):
    def test_sanitize_handles_none_and_empty(self):
        self.assertEqual(common.sanitize(None), "")
        self.assertEqual(common.sanitize(""), "")
        self.assertEqual(common.sanitize(ValueError("")), "")
        self.assertEqual(common.sanitize(Exception()), "")

    def test_sanitize_redacts_paths_hosts_credentials(self):
        self.assertNotIn("alice", common.sanitize("open /Users/alice/dev/x failed"))
        self.assertIn("<path>", common.sanitize("open /Users/alice/dev/x failed"))
        self.assertIn("<host>", common.sanitize("cannot reach buildbox.internal"))
        s = common.sanitize("token=SUPERSECRET leftover")
        self.assertNotIn("SUPERSECRET", s)
        self.assertIn("<redacted>", s)

    def test_sanitize_limit(self):
        self.assertEqual(len(common.sanitize("x" * 500, limit=10)), 10)

    def test_v3_error_objects_carry_no_free_text(self):
        exc = ValueError('token="example secret phrase" in /Users/example/x')
        obj = common.err_obj(exc, "timing")
        self.assertEqual(sorted(obj), ["category", "phase", "type"])
        self.assertEqual(obj["category"], "value")
        self.assertNotIn("example", json.dumps(obj))
        legacy = common.err_obj(exc, "timing", version=2)
        self.assertIn("message", legacy)

    def test_packing_failed_carries_attempts_and_category(self):
        exc = common.PackingFailed(["lo", "hi"], ValueError("x"))
        self.assertEqual(exc.attempts, ["lo", "hi"])
        self.assertEqual(common.classify(exc), "packing")

    def test_finite_check_makes_diagnostics_json_safe(self):
        chk = {"stage": "rate_shape", "ok": False, "rel_err": float("nan"),
               "tolerance": 1e-5}
        safe = common.finite_check(chk)
        self.assertIsNone(safe["rel_err"])
        self.assertTrue(safe["non_finite"])
        self.assertFalse(safe["ok"])
        common.write_receipt({"check": safe}, os.path.join(
            tempfile.mkdtemp(), "r.json"))  # must serialize

    def test_parse_int_tuple(self):
        self.assertEqual(common.parse_int_tuple("64x128x4", "tile"), (64, 128, 4))
        for bad in ("64x128", "64x128x4x2", "64x0x4", "64x-1x4", "a x b x c"):
            with self.assertRaises(ValueError):
                common.parse_int_tuple(bad, "tile")

    def test_validate_tile_rejects_zero_and_bad_simdgroups(self):
        self.assertEqual(common.validate_tile((64, 128, 4)), (64, 128, 4))
        for bad in ((0, 128, 4), (64, 0, 4), (64, 128, 3), (64, 128, 0),
                    (64, 128)):
            with self.assertRaises(ValueError):
                common.validate_tile(bad)

    def test_write_receipt_is_strict_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.json")
            common.write_receipt({"a": 1}, path)
            self.assertEqual(json.load(open(path)), {"a": 1})
            with self.assertRaises(ValueError):
                common.write_receipt({"a": float("nan")}, path)
            self.assertEqual(json.load(open(path)), {"a": 1})
            self.assertEqual([p for p in os.listdir(d) if p.startswith(".receipt-")], [])

    def test_load_json_rejects_nan(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.json")
            with open(path, "w") as f:
                f.write('{"a": NaN}')
            with self.assertRaises(ValueError):
                common.load_json(path)

    def test_source_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.py")
            with open(p, "wb") as f:
                f.write(b"abc")
            self.assertEqual(common.source_hashes([p]),
                             {"x.py": hashlib.sha256(b"abc").hexdigest()})

    def test_probe_files_includes_common(self):
        files = common.probe_files(os.path.join(PROBES, "lane_sweep.py"))
        self.assertIn("_common.py", [os.path.basename(f) for f in files])
        self.assertIn("lane_sweep.py", [os.path.basename(f) for f in files])


class TestPackingDiscovery(unittest.TestCase):
    """Packing discovery against the *real* tuple-returning helper contract."""

    def tearDown(self):
        lane_sweep.check_tile = self._orig
        integer_lanes.check_tile = self._orig_int

    def setUp(self):
        self._orig = lane_sweep.check_tile
        self._orig_int = integer_lanes.check_tile

    def _tuple_stub(self, ok_orders, calls=None):
        def stub(lane, tile, shape, rng, order):
            if calls is not None:
                calls.append(order)
            ok = order in ok_orders
            return ok, {"stage": "check_shape", "tile": f"{tile[0]}x{tile[1]}x{tile[2]}",
                        "rel_err": 0.0, "tolerance": 1e-5, "ok": ok}, {}
        return stub

    def test_lane_index_is_canonical_order(self):
        self.assertEqual(lane_sweep.LANE_INDEX,
                         {lane: i for i, lane in enumerate(lane_sweep.LANES)})
        self.assertEqual(integer_lanes.LANES_INDEX,
                         {lane: i for i, lane in enumerate(integer_lanes.LANES)})

    def test_geometry_discover_packing_lo(self):
        calls = []
        lane_sweep.check_tile = self._tuple_stub({"lo"}, calls)
        order, attempts = geometry_sweep.discover_packing(
            "fp4*fp4->f32", (2048, 4096, 4096), None)
        self.assertEqual((order, attempts), ("lo", ["lo"]))
        self.assertEqual(calls, ["lo"])

    def test_geometry_discover_packing_hi_fallback(self):
        calls = []
        lane_sweep.check_tile = self._tuple_stub({"hi"}, calls)
        order, attempts = geometry_sweep.discover_packing(
            "fp4*fp4->f32", (2048, 4096, 4096), None)
        self.assertEqual((order, attempts), ("hi", ["lo", "hi"]))
        self.assertEqual(calls, ["lo", "hi"])

    def test_geometry_discover_packing_both_fail_carries_attempts(self):
        lane_sweep.check_tile = self._tuple_stub(set())
        with self.assertRaises(common.PackingFailed) as ctx:
            geometry_sweep.discover_packing("fp4*fp4->f32", (2048, 4096, 4096), None)
        self.assertEqual(ctx.exception.attempts, ["lo", "hi"])

    def test_geometry_discover_packing_exception_keeps_attempts(self):
        def boom(lane, tile, shape, rng, order):
            raise ValueError("no device")
        lane_sweep.check_tile = boom
        with self.assertRaises(common.PackingFailed) as ctx:
            geometry_sweep.discover_packing("fp4*fp4->f32", (2048, 4096, 4096), None)
        self.assertEqual(ctx.exception.attempts, ["lo"])
        self.assertEqual(common.classify(ctx.exception), "packing")

    def test_integer_discover_packing_contract(self):
        cases = [({"lo"}, ("lo", ["lo"])), ({"hi"}, ("hi", ["lo", "hi"]))]
        for ok_orders, expected in cases:
            integer_lanes.check_tile = lambda lane, tile, relaxed, rng, order, \
                ok_orders=ok_orders: {"ok": order in ok_orders}
            self.assertEqual(integer_lanes.discover_packing(
                "i8*i4b->i32", (64, 128, 4), False, None), expected)
        integer_lanes.check_tile = lambda *a: {"ok": False}
        with self.assertRaises(common.PackingFailed) as ctx:
            integer_lanes.discover_packing("i8*i4b->i32", (64, 128, 4), False, None)
        self.assertEqual(ctx.exception.attempts, ["lo", "hi"])

    def test_geometry_is_packed(self):
        self.assertTrue(geometry_sweep.is_packed("fp4*fp4->f32"))
        self.assertTrue(geometry_sweep.is_packed("f16_mx_fp4->f32"))
        self.assertFalse(geometry_sweep.is_packed("e4m3*e4m3->f32"))


class TestProbeMainToValidator(unittest.TestCase):
    """Drive each probe's `main()` with stubbed GPU seams and validate output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig = {
            "ls_check": lane_sweep.check_tile,
            "ls_time": lane_sweep.time_tile,
            "int_check": integer_lanes.check_tile,
            "int_rate": integer_lanes.rate_check,
            "int_time": integer_lanes.time_tile,
            "int_operands": integer_lanes.operands,
        }

    def tearDown(self):
        lane_sweep.check_tile = self._orig["ls_check"]
        lane_sweep.time_tile = self._orig["ls_time"]
        integer_lanes.check_tile = self._orig["int_check"]
        integer_lanes.rate_check = self._orig["int_rate"]
        integer_lanes.time_tile = self._orig["int_time"]
        integer_lanes.operands = self._orig["int_operands"]

    def _stub_integer_gpu(self):
        integer_lanes.operands = lambda *a, **k: (None, None, None, None)

    def _stub_lane_sweep(self, bad_first_lane=False):
        def check(lane, tile, shape, rng, order):
            if bad_first_lane and shape == lane_sweep.CHK:
                return False, {"stage": "check_shape",
                               "tile": f"{tile[0]}x{tile[1]}x{tile[2]}",
                               "rel_err": float("nan"), "tolerance": 1e-5,
                               "ok": False}, {}
            entry = {"stage": ("check" if shape == lane_sweep.CHK else "rate")
                     + "_shape", "tile": f"{tile[0]}x{tile[1]}x{tile[2]}",
                     "rel_err": 0.0, "tolerance": 1e-5, "ok": True}
            return True, entry, {"kern": None, "inputs": None,
                                 "m": shape[0], "n": shape[2]}
        lane_sweep.check_tile = check
        lane_sweep.time_tile = lambda ctx, tile, reps, warmups: \
            [1.3 + 0.01 * i for i in range(reps)]

    def test_float_subset_without_f16_validates(self):
        # A run measuring only e5m2 is legitimate; the schema no longer demands
        # an f16 lane, and the baseline is required only for ratio cells.
        self._stub_lane_sweep()
        path = os.path.join(self.tmp.name, "lane_sweep.json")
        rc = run_main(self, lane_sweep,
                      ["--json", path, "--lanes", "e5m2*e5m2->f32"])
        self.assertEqual(rc, 0)
        rec = validate_probe_receipt(self, path)
        self.assertEqual(rec["receipt_schema"], 3)
        self.assertNotIn("f16*f16->f32", rec)

    def test_float_non_finite_check_continues_and_serializes(self):
        self._stub_lane_sweep(bad_first_lane=True)
        path = os.path.join(self.tmp.name, "lane_sweep.json")
        rc = run_main(self, lane_sweep, ["--json", path])
        self.assertEqual(rc, 1)  # no lane usable -> nonzero exit
        rec = validate_probe_receipt(self, path)
        self.assertTrue(rec["complete"])
        bad = rec["e4m3*e4m3->f32"]
        self.assertEqual(bad["status"], "wrong")
        check = bad["checks"][0]
        self.assertIsNone(check["rel_err"])
        self.assertTrue(check["non_finite"])
        self.assertFalse(check["ok"])

    def test_integer_packing_failure_is_terminal_evidence(self):
        self._stub_integer_gpu()

        def check(lane, tile, relaxed, rng, order):
            return {"stage": "check_shape", "rel_err": 0.0, "tolerance": "exact",
                    "ok": True}
        integer_lanes.check_tile = check
        integer_lanes.rate_check = lambda lane, tile, relaxed, a, b, am, bm, shape: \
            (True, {"stage": f"rate_shape_{shape[0]}x{shape[1]}x{shape[2]}",
                    "rel_err": 0.0, "tolerance": "exact", "ok": True})
        integer_lanes.time_tile = lambda *a, **k: (0.0139, 0.0139, [13.9] * 7)
        orig_discover = integer_lanes.discover_packing

        def discover(lane, tile, relaxed, rng):
            if lane == "i8*i4b->i32":
                raise common.PackingFailed(["lo", "hi"])
            return "lo", ["lo"]
        integer_lanes.discover_packing = discover
        try:
            path = os.path.join(self.tmp.name, "integer.json")
            rc = run_main(self, integer_lanes, [
                "--json", path, "--lanes", "f16*f16->f32,i8*i4b->i32",
                "--tiles", "64x128x4", "--shapes", "2048x4096x4096,4096x4096x4096"])
        finally:
            integer_lanes.discover_packing = orig_discover
        self.assertEqual(rc, 0)
        rec = validate_probe_receipt(self, path)
        self.assertTrue(rec["complete"])
        self.assertEqual(rec["protocol"]["lanes_requested"],
                         ["f16*f16->f32", "i8*i4b->i32"])
        rows = rec["rows"]
        self.assertEqual(len(rows), 2 * 1 * 2)  # lanes x tiles x shapes
        failed = [r for r in rows if r["status"] == "failed"]
        self.assertEqual(len(failed), 2)
        for row in failed:
            self.assertEqual(row["stage"], "packing_check")
            self.assertEqual(row["packing_attempts"], ["lo", "hi"])
            self.assertEqual(row["N"], 4096)
            self.assertNotIn("message", row["error"])

    def test_integer_subset_success_validates(self):
        self._stub_integer_gpu()
        integer_lanes.check_tile = lambda *a: {
            "stage": "check_shape", "rel_err": 0.0, "tolerance": "exact",
            "ok": True}
        integer_lanes.rate_check = lambda lane, tile, relaxed, a, b, am, bm, shape: \
            (True, {"stage": f"rate_shape_{shape[0]}x{shape[1]}x{shape[2]}",
                    "rel_err": 0.0, "tolerance": "exact", "ok": True})
        integer_lanes.time_tile = lambda *a, **k: (0.0139, 0.0139, [13.9] * 7)
        path = os.path.join(self.tmp.name, "integer.json")
        rc = run_main(self, integer_lanes, [
            "--json", path, "--lanes", "f16*f16->f32", "--tiles", "64x128x4",
            "--shapes", "2048x4096x4096"])
        self.assertEqual(rc, 0)
        rec = validate_probe_receipt(self, path)
        self.assertEqual(len(rec["rows"]), 1)
        self.assertEqual(rec["rows"][0]["M"], 2048)

    def test_geometry_packed_lane_hi_fallback_validates(self):
        calls = []

        def check(lane, tile, shape, rng, order):
            calls.append(order)
            ok = order == "hi"
            entry = {"stage": ("check" if shape == lane_sweep.CHK else "rate")
                     + "_shape", "tile": f"{tile[0]}x{tile[1]}x{tile[2]}",
                     "rel_err": 0.0, "tolerance": 1e-5, "ok": ok}
            return ok, entry, {"kern": None, "inputs": None,
                               "m": shape[0], "n": shape[2]}
        lane_sweep.check_tile = check
        lane_sweep.time_tile = lambda ctx, tile, reps, warmups: \
            [1.3 + 0.01 * i for i in range(reps)]
        path = os.path.join(self.tmp.name, "geometry.json")
        rc = run_main(self, geometry_sweep, [
            "--json", path, "--lanes", "f16*f16->f32,fp4*fp4->f32",
            "--shapes", "2048x4096x4096"])
        self.assertEqual(rc, 0)
        rec = validate_probe_receipt(self, path)
        meta = rec["lane_meta"]["fp4*fp4->f32"]
        self.assertEqual(meta["packing"], "hi")
        self.assertEqual(meta["packing_attempts"], ["lo", "hi"])
        self.assertEqual(rec["lane_meta"]["f16*f16->f32"]["packing_attempts"], [])
        self.assertIn("lo", calls)

    def test_geometry_packing_failure_attempts_are_not_reconstructed(self):
        def check(lane, tile, shape, rng, order):
            raise ValueError("device unavailable")
        lane_sweep.check_tile = check
        path = os.path.join(self.tmp.name, "geometry.json")
        rc = run_main(self, geometry_sweep, [
            "--json", path, "--lanes", "fp4*fp4->f32",
            "--shapes", "2048x4096x4096"])
        self.assertEqual(rc, 1)  # only lane failed -> nonzero exit
        rec = validate_probe_receipt(self, path)
        self.assertTrue(rec["complete"])
        meta = rec["lane_meta"]["fp4*fp4->f32"]
        self.assertEqual(meta["packing_attempts"], ["lo"])  # only lo ran
        self.assertIsNone(meta["packing"])


if __name__ == "__main__":
    unittest.main()
