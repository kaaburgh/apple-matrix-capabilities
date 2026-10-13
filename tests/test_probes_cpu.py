#!/usr/bin/env python3
"""CPU-only tests for probes/_common.py and the probes' pure helpers.

Run from the repository root:

    python -m unittest discover -s tests -v

The probes import `mlx.core` at module level (and call `mx.set_memory_limit`
during import), so these tests install a stub `mlx` module first. Nothing here
touches a GPU: the packing-discovery helpers are exercised with a faked
`check_tile`, and the shared helpers are pure Python.
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
            h = common.source_hashes([p])
            self.assertEqual(h, {"x.py": hashlib.sha256(b"abc").hexdigest()})

    def test_probe_files_includes_common(self):
        files = common.probe_files(os.path.join(PROBES, "lane_sweep.py"))
        self.assertIn("_common.py", [os.path.basename(f) for f in files])
        self.assertIn("lane_sweep.py", [os.path.basename(f) for f in files])


class TestPackingDiscovery(unittest.TestCase):
    """The integer/Legacy packing negative control must be testable without a GPU."""

    def test_lane_index_is_canonical_order(self):
        expected = {lane: i for i, lane in enumerate(lane_sweep.LANES)}
        self.assertEqual(lane_sweep.LANE_INDEX, expected)
        self.assertEqual(integer_lanes.LANES_INDEX,
                         {lane: i for i, lane in enumerate(integer_lanes.LANES)})

    def test_integer_discover_packing_lo(self):
        orig = integer_lanes.check_tile
        integer_lanes.check_tile = lambda *a: {"ok": True}
        try:
            order, attempts = integer_lanes.discover_packing("i8*i4b->i32",
                                                             (64, 128, 4), False, None)
            self.assertEqual((order, attempts), ("lo", ["lo"]))
        finally:
            integer_lanes.check_tile = orig

    def test_integer_discover_packing_hi(self):
        orig = integer_lanes.check_tile
        integer_lanes.check_tile = lambda lane, tile, relaxed, rng, order: {
            "ok": order == "hi"}
        try:
            order, attempts = integer_lanes.discover_packing("i4b*i4b->i32",
                                                             (64, 128, 4), False, None)
            self.assertEqual((order, attempts), ("hi", ["lo", "hi"]))
        finally:
            integer_lanes.check_tile = orig

    def test_integer_discover_packing_failure_raises(self):
        orig = integer_lanes.check_tile
        integer_lanes.check_tile = lambda *a: {"ok": False}
        try:
            with self.assertRaises(RuntimeError):
                integer_lanes.discover_packing("i4b*i4b->i32", (64, 128, 4),
                                               False, None)
        finally:
            integer_lanes.check_tile = orig

    def test_geometry_discover_packing_uses_lane_sweep_check(self):
        orig = lane_sweep.check_tile
        lane_sweep.check_tile = lambda lane, tile, shape, rng, order: {"ok": True}
        try:
            order, attempts = geometry_sweep.discover_packing(
                "e4m3*e4m3->f32", (2048, 4096, 4096), None)
            self.assertEqual((order, attempts), ("lo", ["lo"]))
            lane_sweep.check_tile = lambda lane, tile, shape, rng, order: {
                "ok": order == "hi"}
            order, attempts = geometry_sweep.discover_packing(
                "fp4*fp4->f32", (2048, 4096, 4096), None)
            self.assertEqual((order, attempts), ("hi", ["lo", "hi"]))
        finally:
            lane_sweep.check_tile = orig

    def test_geometry_is_packed(self):
        self.assertTrue(geometry_sweep.is_packed("fp4*fp4->f32"))
        self.assertTrue(geometry_sweep.is_packed("f16_mx_fp4->f32"))
        self.assertFalse(geometry_sweep.is_packed("e4m3*e4m3->f32"))


if __name__ == "__main__":
    unittest.main()
