"""CPU-only contract tests for the disposable Phase 0b protocol.

No timings or hardware facts from these tests represent Apple GPU results.
"""

import importlib.util
from io import StringIO
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "phase0b_baseline", ROOT / "tools/phase0b_baseline.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TestGuardsAndStatistics(unittest.TestCase):
    def test_capture_debug_modes_refused(self):
        for name in mod.PROHIBITED_ENV:
            with self.subTest(name=name):
                with self.assertRaises(mod.BlockedError):
                    mod.validate_timing_env({name: "1"})
        mod.validate_timing_env({name: "0" for name in mod.PROHIBITED_ENV})

    def test_environment_redacted_with_buffer_limits_visible(self):
        values = mod.safe_environment({
            "METAL_DEVICE_WRAPPER_TYPE": "/Users/alice/dev/metal",
            "MLX_MAX_OPS_PER_BUFFER": "120",
            "MLX_MAX_MB_PER_BUFFER": "64",
            "MLX_ENABLE_TF32": "1",
        })
        self.assertNotIn("alice", repr(values))
        self.assertEqual(values["METAL_DEVICE_WRAPPER_TYPE"], "set")
        self.assertEqual(values["MLX_MAX_OPS_PER_BUFFER"], "120")
        self.assertEqual(values["MLX_MAX_MB_PER_BUFFER"], "64")

    def test_memory_budget_and_tile_validation(self):
        choices, omitted = mod.eligible_ns(32 * mod.MIB, 512 * mod.MIB)
        self.assertEqual(choices, [1, 2, 4, 8, 16])
        self.assertFalse(omitted)
        choices, omitted = mod.eligible_ns(32 * mod.MIB, 100 * mod.MIB)
        self.assertEqual(choices, [1, 2])
        self.assertIn("4", omitted)
        with self.assertRaises(ValueError):
            mod.eligible_ns(0, 512 * mod.MIB)
        with self.assertRaisesRegex(ValueError, "does not divide"):
            mod.validate_tile_shape((64, 1024, 256), (128, 128, 8))
        mod.validate_tile_shape((2048, 4096, 4096), (128, 128, 8))

    def test_window_criterion(self):
        self.assertFalse(mod.window_converged([1, 1]))
        self.assertTrue(mod.window_converged([5, 1, 1.01, 1.02]))
        self.assertFalse(mod.window_converged([1, 1.1, .9]))
        self.assertFalse(mod.window_converged([1, math.nan, 1]))

    def test_exact_ols_fit_and_repeat_distribution(self):
        obs = [(n, 2.5 + .75*n) for n in mod.NS for _ in range(3)]
        fit = mod.fit_tn(obs)
        self.assertAlmostEqual(fit["a_effective_fixed_ms"], 2.5)
        self.assertAlmostEqual(fit["b_back_to_back_ms_per_dispatch"], .75)
        self.assertAlmostEqual(fit["b_ols_standard_error_ms"], 0)
        self.assertIn("secondary", fit["warning"].lower())
        self.assertEqual(mod.quantile_sorted([0, 2, 4], .5), 2)
        with self.assertRaises(ValueError):
            mod.fit_tn([(1, 1), (1, 2), (2, 3), (2, 4)])
        with self.assertRaises(ValueError):
            mod.fit_tn([(1, math.inf), (2, 2), (4, 3)])
        with self.assertRaises(ValueError):
            mod.data_summary([0, 1])
        stats = mod.data_summary([1, 2, 4])
        self.assertEqual(stats["median_ms"], 2)
        self.assertEqual(stats["spread_max_over_min"], 4)

    def test_timed_batch_splits_lazy_construction_from_evaluation(self):
        outputs, calls = [], []
        def f():
            result = object()
            outputs.append(result)
            return result
        def sync(*arrays):
            calls.append(arrays)
        result = mod.timed_batch(f, sync, 4)
        self.assertEqual(tuple(outputs), calls[0])
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(result["construct_ms"], 0)
        self.assertGreaterEqual(result["eval_ms"], 0)

    def test_per_repeat_slopes_not_just_pooled_se(self):
        def observation(_, __, n):
            # deterministic 2 ms fixed + 0.75 ms/dispatch eval, 0.2 ms
            # construction / dispatch. Construction must NOT enter fit.
            return {"eval_ms": 2 + .75*n, "construct_ms": .2*n}
        with patch.object(mod, "timed_batch", side_effect=observation):
            result = mod.measure_tn(lambda: None, lambda *xs: None,
                                    [1, 2, 4, 8, 16], repetitions=7)
        self.assertEqual(len(result["per_repetition_fits"]), 7)
        self.assertEqual([r["order"] for r in result["per_repetition_fits"]],
                         ["asc", "desc", "asc", "desc", "asc", "desc", "asc"])
        self.assertEqual(result["b_repetition_distribution"]["repetitions_by_order"],
                         {"asc": 4, "desc": 3})
        self.assertAlmostEqual(
            result["b_repetition_distribution"]["median_by_order_ms_per_dispatch"]["asc"], .75)
        self.assertAlmostEqual(
            result["b_repetition_distribution"]["median_by_order_ms_per_dispatch"]["desc"], .75)
        self.assertAlmostEqual(
            result["b_repetition_distribution"]["order_gap_desc_minus_asc_ms_per_dispatch"], 0.0)
        self.assertAlmostEqual(
            result["b_repetition_distribution"]["median_ms_per_dispatch"], .75)
        self.assertAlmostEqual(
            result["fit_pooled_secondary"]["b_back_to_back_ms_per_dispatch"], .75)
        self.assertEqual(result["per_n"]["4"]["construct_ms"]["median_ms"], .8)
        self.assertTrue(all("offset_s" in x for x in result["chronological"]))
        self.assertEqual([x["n"] for x in result["chronological"][:10]],
                         [1, 2, 4, 8, 16, 16, 8, 4, 2, 1])

    def test_per_repeat_order_gap_detects_scripted_drift(self):
        count = [0]
        def observation(_, __, n):
            rep = count[0] // 5
            count[0] += 1
            slope = 1.0 if rep % 2 == 0 else 2.0
            return {"eval_ms": 3.0 + slope*n, "construct_ms": 0.25*n}
        with patch.object(mod, "timed_batch", side_effect=observation):
            result = mod.measure_tn(lambda: None, lambda *xs: None,
                                    [1, 2, 4, 8, 16], repetitions=6)
        distribution = result["b_repetition_distribution"]
        self.assertAlmostEqual(distribution["median_by_order_ms_per_dispatch"]["asc"], 1.0)
        self.assertAlmostEqual(distribution["median_by_order_ms_per_dispatch"]["desc"], 2.0)
        self.assertAlmostEqual(distribution["order_gap_desc_minus_asc_ms_per_dispatch"], 1.0)

    def test_interleaved_abba_tracks_evaluation_and_construction(self):
        def a():
            return object()
        def b():
            return object()
        with patch.object(mod, "timed_batch", side_effect=lambda f, sync, n:
                          {"eval_ms": 2.0 if f is b else 1.0,
                           "construct_ms": .1}):
            r = mod.interleaved_ab(a, b, lambda *xs: None, n=4, repetitions=4)
        self.assertEqual([x["tile"] for x in r["chronological"]],
                         ["A", "B", "B", "A", "A", "B", "B", "A"])
        self.assertEqual(r["median_ratio_B_over_A_eval"], 2)
        self.assertEqual(len(r["A"]["eval_ms"]["samples_ms"]), 4)
        self.assertTrue(all("offset_s" in x for x in r["chronological"]))

    def test_bounded_warmup_settles_and_preserves_first_batches(self):
        class FakeClock:
            t = 0.0
            def now(self):
                self.t += .05
                return self.t
        clock = FakeClock()
        calls = iter([5, 4, 3, 2] + [1.0]*200)
        def sample(*args):
            return {"eval_ms": next(calls), "construct_ms": .1}
        with patch.object(mod.time, "perf_counter", side_effect=clock.now):
            with patch.object(mod, "timed_batch", side_effect=sample):
                result = mod.warmup(lambda: None, lambda *xs: None, n=4,
                                    min_s=.4, max_s=2.0, window_s=.2,
                                    tolerance=.02)
        self.assertTrue(result["converged"])
        self.assertGreaterEqual(result["elapsed_s"], .4)
        self.assertGreater(len(result["raw_batches"]), 3)
        self.assertGreater(result["raw_batches"][0]["eval_ms"],
                           result["raw_batches"][-1]["eval_ms"])

    def test_bounded_warmup_exhaustion_never_converges(self):
        class FakeClock:
            t = 0.0
            def now(self):
                self.t += .05
                return self.t
        clock = FakeClock()
        count = [0]
        def sample(*args):
            count[0] += 1
            return {"eval_ms": float(count[0]), "construct_ms": .1}
        with patch.object(mod.time, "perf_counter", side_effect=clock.now):
            with patch.object(mod, "timed_batch", side_effect=sample):
                result = mod.warmup(lambda: None, lambda *xs: None, n=4,
                                    min_s=.4, max_s=.8, window_s=.2,
                                    tolerance=.02)
        self.assertFalse(result["converged"])
        self.assertEqual(result["criterion"], "max_warmup_budget_exhausted")


class TestSyntheticRun(unittest.TestCase):
    def setUp(self):
        self.context = types.SimpleNamespace(kern="fake", inputs=[], m=2048, n=4096)
        core = types.ModuleType("mlx.core")
        core.device_info = lambda: {"memory_size": 32 * 2**30,
                                    "architecture": "synthetic"}
        core.eval = lambda *args: None
        mlx = types.ModuleType("mlx")
        mlx.core = core
        lane = types.ModuleType("lane_sweep")
        lane.LANES = {"f16*f16->f32": True}
        lane.SEED = 20261012
        lane.LANE_INDEX = {"f16*f16->f32": 5}
        lane.CHK = (64, 1024, 256)
        lane.RATE = (2048, 4096, 4096)
        lane.TILES = [(64,128,4), (64,64,2), (64,64,4),
                      (64,32,4), (128,128,8)]
        lane.lane_relaxed = lambda lane: True
        lane.call = lambda *args: object()
        lane.check_tile = Mock(side_effect=self.check)
        lane.time_tile = Mock(return_value=[5,4,3,2,1])
        lane.C = types.SimpleNamespace(
            finite_check=lambda x: x,
            gpu_cores=lambda: "unknown",
            first_line=lambda s: s or "unknown",
            run_text=lambda *args: "",
            toolchain_meta=lambda: {"xcode": "synthetic"},
            power_meta=lambda: {"source": "unknown"})
        self.lane = lane
        self.modules = patch.dict(sys.modules, {
            "mlx": mlx, "mlx.core": core, "lane_sweep": lane,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {x: "0" for x in mod.PROHIBITED_ENV})
        self.env.start()
        self.addCleanup(self.env.stop)
        patches = [
            patch.object(mod.platform, "system", return_value="Darwin"),
            patch.object(mod.platform, "machine", return_value="arm64"),
            patch.object(mod, "warmup", return_value={
                "converged": True, "elapsed_s": 1, "raw_batches": []}),
            patch.object(mod, "measure_tn", return_value={
                "per_repetition_fits": [], "fit_pooled_secondary": {}}),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def check(self, lane, tile, shape, rng, packing, stage=None):
        return True, {"rel_err": 0.0, "ok": True, "stage": stage}, {
            "kern": "synthetic", "inputs": [], "m": shape[0], "n": shape[2]}

    def args(self, **kwargs):
        values = dict(lane="f16*f16->f32", tile_a=0, tile_b=None,
                      shape="rate", seed=20261012, reps=7, output_budget_mib=512,
                      warm_batch=8, min_warmup_s=.5, max_warmup_s=8,
                      window_s=.2, tolerance=.02, upstream_replica=False)
        values.update(kwargs)
        return types.SimpleNamespace(**values)

    def test_rate_check_uses_canonical_then_requested_tile(self):
        result = mod.run(self.args(tile_a=4, upstream_replica=True))
        calls = self.lane.check_tile.call_args_list
        self.assertEqual([c.args[1] for c in calls], [self.lane.TILES[0],
                                                       self.lane.TILES[4]])
        self.assertIs(calls[0].args[3], calls[1].args[3])
        self.assertEqual(result["upstream_protocol_replica"]["samples_ms"],
                         [5,4,3,2,1])
        self.assertTrue(result["steady_state_claim_allowed"])
        json.dumps(result, allow_nan=False)

    def test_check_shape_rejects_zero_threadgroup_tile(self):
        with self.assertRaisesRegex(mod.PhaseFailure, "does not divide"):
            mod.run(self.args(shape="check", tile_a=4))
        self.lane.check_tile.assert_not_called()

    def test_failed_precheck_never_reaches_timing(self):
        self.lane.check_tile.return_value = (False, {"ok":False}, {})
        self.lane.check_tile.side_effect = None
        with patch.object(mod, "measure_tn") as measure:
            with self.assertRaises(mod.PhaseFailure):
                mod.run(self.args())
            measure.assert_not_called()

    def test_budget_guard_before_warmup(self):
        with patch.object(mod, "warmup") as warm:
            with self.assertRaises(mod.PhaseFailure) as exc:
                mod.run(self.args(output_budget_mib=32))
            self.assertEqual(exc.exception.stage, "budget")
            warm.assert_not_called()


class TestPersistence(unittest.TestCase):
    def test_serialization_failure_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "new.json"
            with self.assertRaises(ValueError):
                mod.store_exclusive_json(dest, {"not_finite": math.inf})
            self.assertFalse(dest.exists())
            mod.store_exclusive_json(dest, {"ok": True})
            with self.assertRaises(FileExistsError):
                mod.store_exclusive_json(dest, {"ok": False})
            self.assertEqual(json.loads(dest.read_text()), {"ok": True})

    def test_errors_write_blocked_or_failed_json_into_temp_project_root(self):
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td)
            (temp_root / "tools").mkdir()
            (temp_root / "experiments" / "test").mkdir(parents=True)
            with patch.object(mod, "__file__", str(temp_root / "tools" / "phase0b_baseline.py")):
                with patch.object(mod, "run", side_effect=mod.PhaseFailure(
                    "budget", mod.BlockedError("not enough memory"))):
                    with patch("sys.stdout", new_callable=StringIO):
                        with patch("sys.stderr", new_callable=StringIO):
                            dest = temp_root / "experiments" / "test" / "fail.json"
                            code = mod.main(["--output", str(dest)])
            self.assertEqual(code, 2)
            report = json.loads(dest.read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["stage"], "budget")
            self.assertEqual(report["error"]["type"], "BlockedError")

    def test_refuses_outside_and_old_evidence_without_modifying_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"tools").mkdir()
            (root/"experiments"/"test").mkdir(parents=True)
            good = root/"experiments"/"test"/"old.json"
            good.write_text("historic")
            with patch.object(mod, "__file__", str(root/"tools"/"phase0b_baseline.py")):
                for path in (good, root/"wrong.json"):
                    with patch("sys.stderr", new_callable=StringIO):
                        with patch("sys.stdout", new_callable=StringIO):
                            status = mod.main(["--output", str(path)])
                    self.assertEqual(status, 2)
            self.assertEqual(good.read_text(), "historic")


if __name__ == "__main__":
    unittest.main()
