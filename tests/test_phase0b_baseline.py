"""CPU-only contracts for the disposable Phase 0b methodology.

No results from these tests represent an Apple GPU timing observation.
"""

import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "phase0b_baseline", ROOT / "tools/phase0b_baseline.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestPhase0bGuards(unittest.TestCase):
    def test_refuse_instrumented_timing(self):
        for var in mod.PROHIBITED_ENV:
            with self.subTest(var=var):
                with self.assertRaisesRegex(ValueError, var):
                    mod.validate_timing_env({var: "1"})

    def test_zero_flags_accepted(self):
        mod.validate_timing_env({name: "0" for name in mod.PROHIBITED_ENV})

    def test_do_not_echo_arbitrary_environment_values(self):
        with self.assertRaisesRegex(ValueError, "METAL_DEVICE_WRAPPER_TYPE"):
            mod.validate_timing_env({"METAL_DEVICE_WRAPPER_TYPE": "/Users/private/test"})
        safe = mod.safe_environment({
            "METAL_DEVICE_WRAPPER_TYPE": "/Users/private/test",
            "MLX_ENABLE_TF32": "1",
        })
        self.assertEqual(safe["METAL_DEVICE_WRAPPER_TYPE"], "set")
        self.assertEqual(safe["MLX_ENABLE_TF32"], "1")
        self.assertNotIn("private", str(safe))

    def test_batch_allocation_budget(self):
        ns, skipped = mod.eligible_ns(32 * mod.MIB, 512 * mod.MIB)
        self.assertEqual(ns, [1, 2, 4, 8, 16])
        self.assertEqual(skipped, {})
        ns, skipped = mod.eligible_ns(32 * mod.MIB, 100 * mod.MIB)
        self.assertEqual(ns, [1, 2])
        self.assertEqual(skipped["4"], "outputs_exceed_memory_budget")
        with self.assertRaises(ValueError):
            mod.eligible_ns(0, 512 * mod.MIB)

    def test_convergence_threshold_exact_and_nonfinite(self):
        self.assertFalse(mod.window_converged([1.0, 1.0]))
        self.assertTrue(mod.window_converged([5, 1.00, 1.01, 1.02], .02))
        self.assertFalse(mod.window_converged([1.0, 1.1, 0.9], .02))
        self.assertFalse(mod.window_converged([1.0, float("nan"), 1.0]))

    def test_linear_fit_exact_data(self):
        observations = [(n, 2.5 + 0.75 * n)
                        for n in (1, 2, 4, 8, 16) for _ in range(3)]
        fit = mod.fit_tn(observations)
        self.assertAlmostEqual(fit["a_effective_fixed_ms"], 2.5, places=10)
        self.assertAlmostEqual(fit["b_back_to_back_ms_per_dispatch"], .75, places=10)
        self.assertAlmostEqual(fit["b_standard_error_ms"], 0, places=10)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=10)
        self.assertIn("not single-dispatch", fit["warning"].lower())

    def test_fit_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            mod.fit_tn([(1, 1), (1, 2), (2, 3), (2, 4)])
        with self.assertRaises(ValueError):
            mod.fit_tn([(1, math.inf), (2, 2), (4, 3), (8, 4)])

    def test_summaries_report_spread_not_best_only(self):
        info = mod.data_summary([1.0, 2.0, 4.0])
        self.assertEqual(info["median_ms"], 2.0)
        self.assertEqual(info["spread_max_over_min"], 4.0)
        self.assertEqual(info["samples_ms"], [1.0, 2.0, 4.0])

    def test_reject_output_outside_gitignored_experiments(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(mod, "run", side_effect=AssertionError("should not run")):
                outside = Path(td) / "receipt.json"
                self.assertEqual(mod.main(["--output", str(outside)]), 1)

    def test_never_overwrite_local_experiment(self):
        base = ROOT / "experiments"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as td:
            path = Path(td) / "old.json"
            path.write_text("historical")
            with patch.object(mod, "run", side_effect=AssertionError("should not run")):
                self.assertEqual(mod.main(["--output", str(path)]), 1)
            self.assertEqual(path.read_text(), "historical")


class TestPhase0bControlLoops(unittest.TestCase):
    def test_timed_batch_creates_n_outputs_and_synchronizes_once(self):
        outputs = []
        synchronized = []
        def factory():
            outputs.append(object())
            return outputs[-1]
        def sync(*args):
            synchronized.append(args)
        t = mod.timed_batch(factory, sync, 4)
        self.assertGreaterEqual(t, 0)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(synchronized), 1)
        self.assertEqual(synchronized[0], tuple(outputs))

    def test_interleaved_ab_order_and_full_samples(self):
        counters = {"A": 0, "B": 0}
        def a():
            counters["A"] += 1
            return 1
        def b():
            counters["B"] += 1
            return 2
        with patch.object(mod, "timed_batch", side_effect=lambda f, sync, n:
                          (2.0 if f is b else 1.0)):
            result = mod.interleaved_ab(a, b, lambda *args: None,
                                        n=4, repetitions=4)
        self.assertEqual([x["tile"] for x in result["chronological"]],
                         ["A", "B", "B", "A", "A", "B", "B", "A"])
        self.assertEqual(result["median_ratio_B_over_A"], 2.0)
        self.assertEqual(len(result["A"]["samples_ms"]), 4)

    def test_tn_sample_order_is_interleaved(self):
        with patch.object(mod, "timed_batch", side_effect=lambda f, sync, n: 1 + n):
            result = mod.measure_tn(lambda: None, lambda *xs: None,
                                     [1, 2, 4], 3)
        self.assertEqual([x["n"] for x in result["chronological"]],
                         [1, 2, 4, 4, 2, 1, 1, 2, 4])
        self.assertEqual(result["fit"]["observations"], 9)
        self.assertEqual(result["fit"]["b_back_to_back_ms_per_dispatch"], 1)

    def test_nonconvergence_is_diagnostic_not_implied(self):
        self.assertFalse(mod.window_converged([1, 2, 3, 4]))
        # This is a synthetic clock/workload; all captured times are mocked.
        ticks = iter([0, .2, .5, .7, 1, 1.2, 1.5, 1.7, 2, 2.2, 2.5, 2.7])
        # Pure nonconvergence gate tested above; no GPU steady-state is inferred.


if __name__ == "__main__":
    unittest.main()
