"""CPU-only contract tests for Phase 0a, independent of working directory."""

import importlib.util
from contextlib import nullcontext
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

    def test_environment_redacts_path_like_values(self):
        with patch.dict(os.environ, {
            "METAL_DEVICE_WRAPPER_TYPE": "/Users/private/metal/debug",
            "MLX_ENABLE_TF32": "1",
        }):
            env = spike.environment()
        self.assertEqual(env["METAL_DEVICE_WRAPPER_TYPE"], "set")
        self.assertEqual(env["MLX_ENABLE_TF32"], "1")
        self.assertNotIn("private", str(env))

    def test_synthetic_success(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"MTL_SHADER_VALIDATION": ""}):
                spike.validate_preflight("Darwin", "arm64", "1", Path(td) / "t.gputrace")


class TestCaptureLifecycle(unittest.TestCase):
    def test_returns_exactly_one_result_and_stops(self):
        mx = Mock()
        mx.gpu = "gpu"
        mx.new_stream.return_value = "fresh_gpu_stream"
        mx.stream.side_effect = lambda stream: nullcontext()
        dispatch = Mock(return_value="result")
        result = spike.capture_once(mx, dispatch, Path("t.gputrace"))
        self.assertEqual(result, "result")
        mx.metal.start_capture.assert_called_once_with("t.gputrace")
        dispatch.assert_called_once()
        mx.eval.assert_called_once_with("result")
        mx.new_stream.assert_called_once_with("gpu")
        mx.stream.assert_called_once_with("fresh_gpu_stream")
        mx.synchronize.assert_not_called()
        mx.metal.stop_capture.assert_called_once()

    def test_stop_on_eval_failure(self):
        mx = Mock()
        mx.gpu = "gpu"
        mx.stream.side_effect = lambda stream: nullcontext()
        mx.eval.side_effect = RuntimeError("synthetic")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            spike.capture_once(mx, lambda: "data", Path("t.gputrace"))
        mx.metal.stop_capture.assert_called_once()

    def test_default_stream_fallback_flushes_before_dispatch(self):
        mx = Mock()
        mx.gpu = "gpu"
        mx.default_stream.return_value = "default"
        sequence = []
        mx.metal.start_capture.side_effect = lambda _: sequence.append("start")
        mx.synchronize.side_effect = lambda _: sequence.append("flush")
        mx.eval.side_effect = lambda _: sequence.append("eval")
        mx.metal.stop_capture.side_effect = lambda: sequence.append("stop")
        result = spike.capture_once(
            mx, lambda: (sequence.append("dispatch") or "data"),
            Path("t.gputrace"), mode="default-stream")
        self.assertEqual(result, "data")
        self.assertEqual(sequence, ["start", "flush", "dispatch", "eval", "stop"])
        mx.new_stream.assert_not_called()

    def test_stop_on_fallback_flush_failure(self):
        mx = Mock()
        mx.gpu = "gpu"
        mx.synchronize.side_effect = RuntimeError("flush failed")
        with self.assertRaisesRegex(RuntimeError, "flush failed"):
            spike.capture_once(mx, lambda: "data", Path("t.gputrace"),
                               mode="default-stream")
        mx.metal.stop_capture.assert_called_once()

    def test_invalid_mode_does_not_start_capture(self):
        mx = Mock()
        with self.assertRaises(ValueError):
            spike.capture_once(mx, lambda: "data", Path("t.gputrace"),
                               mode="unknown")
        mx.metal.start_capture.assert_not_called()

    def test_stream_created_after_capture_started(self):
        order = []
        mx = Mock()
        mx.gpu = "gpu"
        mx.metal.start_capture.side_effect = lambda _: order.append("start")
        mx.new_stream.side_effect = lambda _: (order.append("new_stream") or "fresh")
        mx.stream.side_effect = lambda _: nullcontext()
        def dispatch():
            order.append("dispatch")
            return "array"
        mx.eval.side_effect = lambda _: order.append("eval")
        mx.metal.stop_capture.side_effect = lambda: order.append("stop")
        spike.capture_once(mx, dispatch, Path("t.gputrace"))
        self.assertEqual(order, ["start", "new_stream", "dispatch",
                                 "eval", "stop"])
        mx.synchronize.assert_not_called()

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
        self.mx.gpu = "gpu"
        self.mx.new_stream = Mock(return_value="fresh_gpu_stream")
        self.mx.stream = Mock(side_effect=lambda stream: nullcontext())
        self.mx.synchronize = Mock()
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
        self.ls.lane_relaxed = lambda lane: True
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
            patch.object(spike, "metal_toolchain", return_value="present"),
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
        self.assertTrue(report["captured_equals_precapture_dispatch"])
        self.assertTrue(report["relaxed_precision"])
        self.assertEqual(report["metal_cli"], "present")
        self.assertEqual(report["expected_kernel_name_contains"],
                         "amc_f16_f16__f32_64x1024x256_64_128_4")
        self.assertFalse(report["trace_replayed_in_xcode"])
        self.assertFalse(report["command_buffers_verified"])
        self.assertEqual(report["capture_mode"], "fresh-stream")
        self.assertEqual(report["expected_capture_command_buffers"], 1)
        self.assertFalse(report["counters_collected"])
        self.assertNotIn("converged", report)
        self.mx.metal.start_capture.assert_called_once()

    def test_no_offline_metal_cli_can_still_capture(self):
        with patch.object(spike, "metal_toolchain", return_value="missing"):
            report = spike.run(self.trace, "check")
        self.assertEqual(report["status"], "capture_written_replay_not_verified")
        self.assertEqual(report["metal_cli"], "missing")

    def test_fallback_mode_is_recorded(self):
        report = spike.run(self.trace, "check", capture_mode="default-stream")
        self.assertEqual(report["status"], "capture_written_replay_not_verified")
        self.assertEqual(report["capture_mode"], "default-stream")
        self.mx.default_stream.assert_called_once()
        self.mx.synchronize.assert_called_once()

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
        calls = self.ls.check_tile.call_args_list
        self.assertEqual([c.args[2] for c in calls], [self.ls.CHK, self.ls.RATE])
        self.assertIs(calls[0].args[3], calls[1].args[3])
        self.assertEqual([c.args[4] for c in calls], ["lo", "lo"])
        self.assertTrue(all(c.args[1] == self.ls.TILES[0] for c in calls))
        self.assertEqual(report["expected_kernel_name_contains"],
                         "amc_f16_f16__f32_2048x4096x4096_64_128_4")

    def test_cli_returns_distinct_blocked_exit_and_json(self):
        from io import StringIO
        with patch.object(spike, "run", return_value={"status": "blocked"}):
            with patch("sys.stdout", new_callable=StringIO) as buf:
                code = spike.main(["--shape", "check"])
        self.assertEqual(code, 2)
        self.assertIn('"blocked"', buf.getvalue())


if __name__ == "__main__":
    unittest.main()
