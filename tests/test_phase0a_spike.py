"""CPU-only contract tests for Phase 0a, independent of working directory."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase0a_capture_spike", ROOT / "tools/phase0a_capture_spike.py")
spike = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spike
SPEC.loader.exec_module(spike)


class TestCaptureSpikePreflight(unittest.TestCase):
    def test_requires_apple_silicon(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(spike.PreflightError, "Apple Silicon"):
                spike.validate_preflight("Linux", "x86_64", "1",
                                         Path(td) / "t.gputrace")

    def test_requires_capture_env(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(spike.PreflightError, "MTL_CAPTURE_ENABLED"):
                spike.validate_preflight("Darwin", "arm64", None,
                                         Path(td) / "t.gputrace")

    def test_rejects_shader_validation(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"MTL_SHADER_VALIDATION": "1"}):
                with self.assertRaisesRegex(spike.PreflightError, "SHADER_VALIDATION"):
                    spike.validate_preflight("Darwin", "arm64", "1",
                                             Path(td) / "t.gputrace")

    def test_rejects_existing_directory_and_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            existing = Path(td) / "t.gputrace"
            existing.mkdir()
            with self.assertRaisesRegex(spike.PreflightError, "already exists"):
                spike.validate_preflight("Darwin", "arm64", "1", existing)
            link = Path(td) / "l.gputrace"
            link.symlink_to(existing)
            with self.assertRaisesRegex(spike.PreflightError, "already exists"):
                spike.validate_preflight("Darwin", "arm64", "1", link)

    def test_rejects_wrong_suffix_and_missing_parent(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(spike.PreflightError, "end in"):
                spike.validate_preflight("Darwin", "arm64", "1", Path(td) / "t.json")
            with self.assertRaisesRegex(spike.PreflightError, "parent"):
                spike.validate_preflight("Darwin", "arm64", "1",
                                         Path(td) / "absent/t.gputrace")

    def test_synthetic_success(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"MTL_SHADER_VALIDATION": ""}):
                spike.validate_preflight("Darwin", "arm64", "1", Path(td) / "t.gputrace")


class TestCaptureLifecycle(unittest.TestCase):
    def test_returns_exactly_one_result_and_stops(self):
        mx = Mock()
        dispatch = Mock(return_value="result")
        result = spike.capture_once(mx, dispatch, Path("t.gputrace"))
        self.assertEqual(result, "result")
        mx.metal.start_capture.assert_called_once_with("t.gputrace")
        dispatch.assert_called_once()
        mx.eval.assert_called_once_with("result")
        mx.metal.stop_capture.assert_called_once()

    def test_stop_on_eval_failure(self):
        mx = Mock()
        mx.eval.side_effect = RuntimeError("synthetic")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            spike.capture_once(mx, lambda: "data", Path("t.gputrace"))
        mx.metal.stop_capture.assert_called_once()

    def test_does_not_stop_when_start_fails(self):
        mx = Mock()
        mx.metal.start_capture.side_effect = RuntimeError("start failed")
        with self.assertRaisesRegex(RuntimeError, "start failed"):
            spike.capture_once(mx, lambda: "data", Path("t.gputrace"))
        mx.metal.stop_capture.assert_not_called()


class TestRunContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = Path(self.tmp.name) / "test.gputrace"

        self.mx = types.ModuleType("mlx.core")
        self.mx.metal = Mock()
        self.mx.eval = Mock()
        self.mx.clear_cache = Mock()
        self.mx.device_info = Mock(return_value={"device_name": "Mock GPU"})
        mlx = types.ModuleType("mlx")
        mlx.core = self.mx

        self.ls = types.ModuleType("lane_sweep")
        self.ls.TILES = [(64, 128, 4)]
        self.ls.CHK = (64, 1024, 256)
        self.ls.RATE = (2048, 4096, 4096)
        self.ls.SEED = 20261012
        self.ls.LANE_INDEX = {"f16*f16->f32": 5}
        self.ls.C = types.SimpleNamespace(finite_check=lambda entry: entry)
        self.ls.check_tile = Mock(side_effect=self.good_check)
        self.ls.call = Mock(return_value=np.array([3, 4], dtype=np.int32))

        self.env = patch.dict(os.environ, {
            "MTL_CAPTURE_ENABLED": "1", "MTL_SHADER_VALIDATION": ""
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.modules = patch.dict(sys.modules, {
            "mlx": mlx, "mlx.core": self.mx, "lane_sweep": self.ls
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        patches = [
            patch.object(spike.platform, "system", return_value="Darwin"),
            patch.object(spike.platform, "machine", return_value="arm64"),
            patch.object(spike, "xcode_version", return_value="Xcode 27.0\nBuild version TEST"),
            patch.object(spike, "metal_toolchain"),
            patch.object(spike, "_machine_metadata", return_value={"sources": {}})
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.mx.metal.start_capture.side_effect = lambda path: Path(path).mkdir()

    @staticmethod
    def good_check(lane, tile, shape, rng, order, stage=None):
        return True, {"ok": True, "stage": stage, "rel_err": 0.0}, {
            "kern": "kernel", "inputs": [], "m": shape[0], "n": shape[2]
        }

    def test_bad_check_does_not_capture(self):
        self.ls.check_tile.return_value = (False, {"ok": False, "rel_err": None}, {})
        self.ls.check_tile.side_effect = None
        report = spike.run(self.trace, "check")
        self.assertEqual((report["status"], report["stage"]), ("failed", "check"))
        self.mx.metal.start_capture.assert_not_called()

    def test_nonfinite_check_does_not_capture(self):
        self.ls.check_tile.side_effect = None
        self.ls.check_tile.return_value = (
            False, {"ok": False, "rel_err": None, "non_finite": True}, {})
        report = spike.run(self.trace, "check")
        self.assertEqual(report["status"], "failed")
        self.mx.metal.start_capture.assert_not_called()

    def test_build_failure_does_not_capture(self):
        self.ls.check_tile.side_effect = RuntimeError("synthetic compile error")
        report = spike.run(self.trace, "check")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "check")
        self.assertEqual(report["error"]["category"], "runtime")
        self.mx.metal.start_capture.assert_not_called()
        self.assertNotIn("synthetic compile error", str(report))

    def test_report_success_does_not_claim_replay_or_na(self):
        report = spike.run(self.trace, "check")
        self.assertEqual(report["status"], "capture_written_replay_not_verified")
        self.assertEqual(report["dispatches_requested"], 1)
        self.assertTrue(report["captured_output_equal"])
        self.assertFalse(report["trace_replayed_in_xcode"])
        self.assertFalse(report["counters_collected"])
        self.assertNotIn("converged", report)
        self.mx.metal.start_capture.assert_called_once()

    def test_missing_trace_is_failed_not_success(self):
        self.mx.metal.start_capture.side_effect = None
        report = spike.run(self.trace, "check")
        self.assertEqual((report["status"], report["stage"]), ("failed", "capture"))

    def test_captured_result_mismatch_is_failed(self):
        self.ls.call.side_effect = [
            np.array([3, 4]),   # pre-capture validation
            np.array([3, 4]),   # warm-up
            np.array([3, 5]),   # actual captured output
        ]
        report = spike.run(self.trace, "check")
        self.assertEqual((report["status"], report["stage"]),
                         ("failed", "captured_output_check"))

    def test_rate_runs_check_then_rate_with_same_rng(self):
        report = spike.run(self.trace, "rate")
        self.assertEqual(report["status"], "capture_written_replay_not_verified")
        self.assertEqual([c.args[2] for c in self.ls.check_tile.call_args_list],
                         [self.ls.CHK, self.ls.RATE])

    def test_cli_returns_distinct_blocked_exit_and_json(self):
        from io import StringIO
        with patch.object(spike, "run", return_value={"status": "blocked"}):
            with patch("sys.stdout", new_callable=StringIO) as buf:
                code = spike.main(["--shape", "check"])
        self.assertEqual(code, 2)
        self.assertIn('"blocked"', buf.getvalue())


if __name__ == "__main__":
    unittest.main()
