"""CPU-only contract tests for reusable PR A capture and live loop.

Mock tests are NOT evidence of GPU capture, Xcode replay, or M5 counters.
"""
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "capture_lane", ROOT / "probes/capture_lane.py")
app = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = app
spec.loader.exec_module(app)


class TestInputGuards(unittest.TestCase):
    def test_modes_are_mutually_exclusive(self):
        p = app.argument_parser()
        with self.assertRaises(SystemExit):
            p.parse_args(["--lane", "f16*f16->f32", "--output", "m.json"])
        with self.assertRaises(SystemExit):
            p.parse_args(["--lane", "f16*f16->f32", "--output", "m.json",
                          "--capture", "a.gputrace", "--loop", "5"])

    def test_paths_reject_receipts_and_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = root / "experiments" / "m3"
            exp.mkdir(parents=True)
            with patch.object(app, "ROOT", root):
                with self.assertRaises(app.B.BlockedError):
                    app.validate_paths(root / "receipts" / "m.json", None)
                with self.assertRaises(app.B.BlockedError):
                    app.validate_paths(exp / "m.json", root / "receipts" / "r.gputrace")
                out, trace = app.validate_paths(exp / "m.json", exp / "r.gputrace")
                self.assertEqual(out, exp / "m.json")
                self.assertEqual(trace, exp / "r.gputrace")
                out.write_text("historical")
                with self.assertRaises(app.B.BlockedError):
                    app.validate_paths(out, None)

    def test_memory_guard(self):
        shape = (2048, 4096, 4096)
        guard = app.allocation_guard(shape, 32 * 2**30, 8, 512)
        self.assertEqual(guard["bytes_per_output"], 32 * 2**20)
        self.assertEqual(guard["batch_output_bytes"], 256 * 2**20)
        with self.assertRaises(app.B.BlockedError):
            app.allocation_guard(shape, 32 * 2**30, 16, 128)

    def test_debug_instrumentation_rejected_in_loop(self):
        lane = types.SimpleNamespace(
            LANES={"f16*f16->f32": ("half",) * 7},
            TILES=[(64, 128, 4)], CHK=(64, 1024, 256), RATE=(2048, 4096, 4096))
        args = app.argument_parser().parse_args(
            ["--loop", "1", "--lane", "f16*f16->f32",
             "--output", "x.json"])
        args.batch = 8
        # Preflight now rejects debug flags in run() BEFORE importing MLX.
        with patch.object(app, "validate_paths",
                          return_value=(Path("m.json"), None)):
            with patch.object(app.platform, "system", return_value="Darwin"):
                with patch.object(app.platform, "machine", return_value="arm64"):
                    report = app.run(args, env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["stage"], "preflight")


class TestCaptureLifecycle(unittest.TestCase):
    def test_one_uses_phase0a_verified_capture(self):
        mx = Mock()
        with patch.object(app.A, "capture_once", return_value="result") as once:
            result = app.capture_batch(mx, lambda: "x", Path("f.gputrace"), 1,
                                       "fresh-stream")
        self.assertEqual(result, ["result"])
        once.assert_called_once()

    def test_multi_batch_uses_one_eval_and_finally_stop(self):
        mx = Mock()
        mx.gpu = "gpu"
        mx.new_stream.return_value = "fresh"
        mx.stream.side_effect = lambda stream: nullcontext()
        factory = Mock(side_effect=["a", "b", "c"])
        values = app.capture_batch(mx, factory, Path("f.gputrace"), 3,
                                   "fresh-stream")
        self.assertEqual(values, ["a", "b", "c"])
        mx.eval.assert_called_once_with("a", "b", "c")
        mx.metal.start_capture.assert_called_once()
        mx.metal.stop_capture.assert_called_once()

    def test_multi_batch_stops_on_eval_error(self):
        mx = Mock()
        mx.gpu = "gpu"
        mx.new_stream.return_value = "fresh"
        mx.stream.side_effect = lambda stream: nullcontext()
        mx.eval.side_effect = RuntimeError("eval error")
        with self.assertRaises(RuntimeError):
            app.capture_batch(mx, lambda: "x", Path("f.gputrace"), 2,
                              "fresh-stream")
        mx.metal.stop_capture.assert_called_once()

    def test_default_stream_preflush_before_dispatch(self):
        mx = Mock()
        mx.gpu = "gpu"
        seq = []
        mx.metal.start_capture.side_effect = lambda *_: seq.append("start")
        mx.default_stream.return_value = "default"
        mx.synchronize.side_effect = lambda *_: seq.append("preflush")
        mx.eval.side_effect = lambda *_: seq.append("eval")
        mx.metal.stop_capture.side_effect = lambda: seq.append("stop")
        app.capture_batch(mx, lambda: (seq.append("dispatch") or "x"),
                          Path("f.gputrace"), 2, "default-stream")
        self.assertEqual(seq, ["start", "preflush", "dispatch", "dispatch",
                               "eval", "stop"])


class TestLaneRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exp = self.root / "experiments" / "m3"
        self.exp.mkdir(parents=True)
        for src in app.SOURCES:
            path = self.root / src
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source\n")
        self.rootPatch = patch.object(app, "ROOT", self.root)
        self.rootPatch.start()
        self.addCleanup(self.rootPatch.stop)

        self.mx = types.ModuleType("mlx.core")
        self.mx.gpu = "gpu"
        self.mx.eval = Mock()
        self.mx.clear_cache = Mock()
        self.mx.metal = Mock()
        self.mx.metal.start_capture.side_effect = lambda p: Path(p).mkdir()
        self.mx.new_stream = Mock(return_value="new_stream")
        self.mx.stream = Mock(side_effect=lambda stream: nullcontext())
        self.mx.device_info = Mock(return_value={"memory_size": 32 * 2**30})
        mlx = types.ModuleType("mlx")
        mlx.core = self.mx

        self.ls = types.ModuleType("lane_sweep")
        self.ls.LANES = {
            "f16*f16->f32": ("half","half","half","half","float",False,False),
            "e4m3*e4m3->f32": ("metal_fp8","uchar","metal_fp8","uchar","float",False,False),
            "fp4*fp4->f32": ("metal_fp4","uchar","metal_fp4","uchar","float",False,False),
            "f32*f32->f32": ("float","float","float","float","float",False,False),
            "strict_f32*f32->f32": ("float","float","float","float","float",False,False),
        }
        self.ls.SEED = 20261012
        self.ls.LANE_INDEX = {lane: i for i, lane in enumerate(self.ls.LANES)}
        self.ls.CHK = (64, 1024, 256)
        self.ls.RATE = (2048, 4096, 4096)
        self.ls.TILES = [(64,128,4), (64,64,2)]
        self.ls.C = types.SimpleNamespace(finite_check=lambda value: value)
        self.ls.lane_kind = lambda dtype: (
            "fp4" if "fp4" in dtype else "f16" if dtype == "half"
            else "f32" if dtype == "float" else "fp8")
        self.ls.lane_relaxed = lambda lane: not lane.startswith("strict_")
        self.ls.check_tile = Mock(side_effect=self.good_check)
        self.ls.call = Mock(return_value=np.array([1,2,3]))
        self.modules = patch.dict(sys.modules, {
            "mlx": mlx, "mlx.core": self.mx, "lane_sweep": self.ls
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

        self.patches = [
            patch.object(app.platform, "system", return_value="Darwin"),
            patch.object(app.platform, "machine", return_value="arm64"),
            patch.object(app.A, "xcode_version", return_value="Xcode 27\nBuild 27A266a"),
            patch.object(app.A, "metal_toolchain", return_value="present"),
            patch.object(app.A, "_machine_metadata", return_value={"device": {"architecture": "synthetic"}}),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def good_check(lane, tile, shape, rng, order, stage=None):
        return True, {"stage": stage, "rel_err": 0.0, "ok": True}, {
            "kern": "fake", "inputs": [], "m": shape[0], "n": shape[2]
        }

    def args(self, **kw):
        settings = {
            "output": self.exp / "result.json",
            "capture": self.exp / "trace.gputrace",
            "loop": None,
            "lane": "f16*f16->f32", "shape": "rate", "tile": 0, "batch": 1,
            "capture_mode": "fresh-stream", "output_budget_mib": 512,
            "min_warmup_s": .5, "max_warmup_s": 1, "window_s": .2,
            "tolerance": .02,
        }
        settings.update(kw)
        return types.SimpleNamespace(**settings)

    def test_capture_manifest_never_claims_convergence_or_replay(self):
        report = app.run(self.args(), env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "capture_written_replay_not_verified")
        self.assertEqual(report["packing"], "lo")
        self.assertEqual(report["dispatches_requested"], 1)
        self.assertNotIn("converged", report)
        self.assertFalse(report["replay"]["verified"])
        self.assertFalse(report["command_buffers_verified"])
        self.assertEqual(set(report["sources"]), set(app.SOURCES))
        self.assertEqual(len(report["checks"]), 2)
        self.assertEqual(self.mx.metal.start_capture.call_count, 1)

    def test_invalid_check_never_captures(self):
        self.ls.check_tile.return_value = (False, {"ok": False}, {})
        self.ls.check_tile.side_effect = None
        report = app.run(self.args(), env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "wrong")
        self.assertEqual(report["stage"], "correctness")
        self.assertEqual(report["lane"], "f16*f16->f32")
        self.assertEqual(report["packing_attempts"], ["lo"])
        self.assertEqual(len(report["checks"]), 1)
        self.assertIn("sources", report)
        self.assertFalse(report["trace_written"])
        self.mx.metal.start_capture.assert_not_called()

    def test_packing_fallback_resets_seed_index(self):
        draws = []
        def conditional(lane, tile, shape, rng, order, stage=None):
            draws.append((order, int(rng.integers(0, 2**31))))
            return (False if order == "lo" else True), {
                "ok": order == "hi", "stage": stage}, {
                    "kern":"fake", "inputs":[], "m":shape[0], "n":shape[2]}
        self.ls.check_tile.side_effect = conditional
        report = app.run(self.args(lane="fp4*fp4->f32", shape="check"),
                         env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "capture_written_replay_not_verified")
        self.assertEqual(report["packing"], "hi")
        self.assertEqual(report["packing_attempts"], ["lo", "hi"])
        self.assertEqual(draws[0][1], draws[1][1])

    def test_rate_failure_never_retries_fp4_packing(self):
        def fail_rate(lane, tile, shape, rng, order, stage=None):
            ok = stage == "check_shape"
            return ok, {"ok": ok, "rel_err": 0 if ok else 1,
                        "stage": stage}, {"kern": "fake", "inputs": [],
                                           "m": shape[0], "n": shape[2]}
        self.ls.check_tile.side_effect = fail_rate
        report = app.run(self.args(lane="fp4*fp4->f32"),
                         env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "wrong")
        self.assertEqual(report["packing_attempts"], ["lo"])
        self.assertEqual([c["stage"] for c in report["checks"]],
                         ["check_shape", "rate_shape"])
        self.mx.metal.start_capture.assert_not_called()

    def test_f16_check_failure_never_retries_hi(self):
        self.ls.check_tile.side_effect = lambda lane, tile, shape, rng, order, stage=None: (
            False, {"ok": False, "rel_err": 2, "stage": stage},
            {"kern": None})
        result = app.run(self.args(), env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(result["packing_attempts"], ["lo"])
        self.assertEqual(result["status"], "wrong")

    def test_converged_warmup_does_not_mask_loop_ramp(self):
        events = []
        ts = iter([0, 0, 0, 100000000, 100000000, 200000000,
                   200000000, 300000000, 300000000, 400000000, 400000000])
        def fake_warm(*args, **kwargs):
            events.append("warmup")
            return {"converged": True, "elapsed_s": .7, "raw_batches": []}
        def fake_sleep(*args):
            events.append("idle")
        with patch.object(app.B, "warmup", side_effect=fake_warm):
            with patch.object(app.B, "timed_batch",
                              side_effect=[{"eval_ms": v, "construct_ms": .01}
                                           for v in (2., 1., 1., 1.)]):
                with patch.object(app.time, "sleep", side_effect=fake_sleep):
                    with patch.object(app.time, "perf_counter_ns",
                                      side_effect=lambda: next(ts)):
                        report = app.loop_batches(
                            lambda: None, lambda *args: None, .35, 1,
                            {"min_s": .5, "max_s": 1.,
                             "window_s": .1, "tolerance": .02})
        self.assertFalse(report["steady_state_claim_allowed"])
        self.assertTrue(report["warmup_converged"])
        self.assertFalse(report["loop_stability"]["stable"])
        self.assertIsNone(report["loop"]["median_eval_ms_per_dispatch_stable_only"])
        self.assertEqual(events, ["idle", "warmup", "idle"])

    def test_long_stationary_noisy_loop_is_not_length_biased(self):
        import math
        # 30 s of a stationary 16 ms batch with ±2% deterministic jitter:
        # old all-0.2s-window max/min would fail on long noisy traces.
        samples = [
            {"offset_s": i * .016, "eval_ms": 16 * (1 + .02 * math.sin(i * .61)),
             "construct_ms": .1, "n": 8}
            for i in range(1875)
        ]
        result = app.loop_stability(samples, batch=8, window_s=.2,
                                    tolerance=.02, seconds_elapsed=30.0)
        self.assertTrue(result["stable"])
        self.assertEqual(len(result["segments"]), 10)
        self.assertGreater(len(result["windows"]), 100)

    def test_isolated_slow_first_batch_not_diluted_into_segments(self):
        samples = [
            {"offset_s": i * .002, "eval_ms": 2.0 if i == 0 else 1.0,
             "construct_ms": .5, "n": 1}
            for i in range(1000)
        ]
        result = app.loop_stability(samples, batch=1, window_s=.2,
                                    tolerance=.02, seconds_elapsed=2.0)
        self.assertTrue(result["segments_populated"])
        self.assertLess(result["segment_range_over_median"], .02)
        self.assertTrue(result["first_batch_spike"])
        self.assertEqual(result["first_batch_over_rest_median"], 2.0)
        self.assertFalse(result["stable"])

    def test_stable_host_eval_not_proof_of_gpu_busy(self):
        # Batches synchronously evaluate in 1 ms each, while graph
        # construction can account for 199 ms. Three timing segments appear
        # stable, but the runner has no Instruments GPU-idle measurement.
        warm = {"converged": True, "elapsed_s": .5, "raw_batches": []}
        ticks = iter([0, 10_000_000, 10_000_000, 210_000_000,
                      210_000_000, 410_000_000, 410_000_000,
                      600_000_000, 600_000_000])
        with patch.object(app.B, "warmup", return_value=warm):
            with patch.object(app.B, "timed_batch", return_value={
                    "eval_ms": 1.0, "construct_ms": 199.0}):
                with patch.object(app.time, "sleep"):
                    with patch.object(app.time, "perf_counter_ns",
                                      side_effect=lambda: next(ticks)):
                        result = app.loop_batches(
                            lambda: None, lambda *a: None, .5, 1,
                            {"min_s": .5, "max_s": 1.,
                             "window_s": .2, "tolerance": .02})
        self.assertTrue(result["host_timing_converged"])
        self.assertFalse(result["steady_state_claim_allowed"])
        self.assertFalse(result["loop"]["host_eval_duty_precheck_passed"])
        self.assertLess(result["loop"]["host_eval_duty_fraction"], .01)
        self.assertFalse(result["loop"]["gpu_idle_fraction_verified"])
        self.assertIsNone(result["loop"]["gpu_idle_fraction"])

    def test_bad_warmup_config_blocks_before_reference_or_import(self):
        bad = self.args(capture=None, loop=1., window_s=2.,
                        min_warmup_s=.5, max_warmup_s=8.)
        result = app.run(bad, env={})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stage"], "preflight")
        self.ls.check_tile.assert_not_called()
        self.mx.device_info.assert_not_called()

    def test_wrong_result_skips_local_traceback(self):
        import io
        from contextlib import redirect_stderr
        self.ls.check_tile.side_effect = lambda *a, **kw: (
            False, {"ok": False, "stage": "check_shape"}, {})
        private_stderr = io.StringIO()
        with redirect_stderr(private_stderr):
            report = app.run(self.args(), env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "wrong")
        self.assertEqual(private_stderr.getvalue(), "")

    def test_loop_warmup_converged_but_ramp_is_not_steady(self):
        samples = [{"offset_s": i * .1, "eval_ms": v, "n": 1}
                   for i, v in enumerate((2.0, 1.0, 1.0, 1.0))]
        stable = app.loop_stability(samples, batch=1, window_s=.1, tolerance=.02)
        self.assertFalse(stable["stable"])
        self.assertEqual(len(stable["windows"]), 4)

    def test_loop_late_throttling_rejected(self):
        samples = [{"offset_s": i * .1, "eval_ms": v, "n": 1}
                   for i, v in enumerate((1.0, 1.0, 1.0, 1.8))]
        self.assertFalse(app.loop_stability(samples, 1, .1, .02)["stable"])
        self.assertTrue(app.loop_stability([
            {"offset_s": i*.1, "eval_ms": 1.0, "n": 1} for i in range(4)],
            1, .1, .02)["stable"])

    def test_blocked_existing_manifest_exits_two_before_run(self):
        self.args().output.write_text("old")
        with patch.object(app, "run") as runner:
            code = app.main(["--lane", "f16*f16->f32", "--loop", "1",
                             "--output", str(self.args().output)])
        self.assertEqual(code, 2)
        runner.assert_not_called()

    def test_failure_after_trace_creation_preserves_trace_status(self):
        self.mx.eval.side_effect = None
        self.mx.metal.start_capture.side_effect = lambda p: Path(p).mkdir()
        self.ls.call.side_effect = [
            np.array([1,2,3]), np.array([1,2,3]),
            np.array([9,9,9])]
        report = app.run(self.args(), env={"MTL_CAPTURE_ENABLED": "1"})
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["trace_written"])
        self.assertEqual(report["trace_basename"], "trace.gputrace")
        self.assertIn("sources", report)

    def test_loop_nonconverged_is_diagnostic_not_steady_state(self):
        warm = {"converged": False, "elapsed_s": 1, "raw_batches": []}
        args = self.args(capture=None, loop=2.0)
        with patch.object(app.B, "warmup", return_value=warm):
            with patch.object(app.B, "timed_batch", return_value={
                "eval_ms":1.0, "construct_ms":.2}):
                with patch.object(app.time, "sleep"):
                    times = iter([0, 100000, 3000000000, 3100000000, 3200000000])
                    with patch.object(app.time, "perf_counter_ns",
                                      side_effect=lambda: next(times)):
                        report = app.run(args, env={})
        self.assertEqual(report["status"], "observed_nonconverged_diagnostic")
        self.assertFalse(report["steady_state_claim_allowed"])
        self.assertEqual(report["loop"]["batch"], 1)
        self.assertEqual(report["loop"]["gpu_idle_fraction"], None)
        json.dumps(report, allow_nan=False)

    def test_main_publishes_manifest_after_capture_created_trace(self):
        trace = self.exp / "new-trace.gputrace"
        output = self.exp / "new-manifest.json"
        def successful_capture(args):
            trace.mkdir()
            return {"status": "capture_written_replay_not_verified",
                    "mode": "capture", "not_upstream_receipt": True}
        with patch.object(app, "run", side_effect=successful_capture):
            rc = app.main(["--lane", "f16*f16->f32",
                           "--capture", str(trace),
                           "--output", str(output)])
        self.assertEqual(rc, 0)
        self.assertTrue(trace.is_dir())
        self.assertEqual(json.loads(output.read_text())["status"],
                         "capture_written_replay_not_verified")

    def test_manifest_writer_does_not_overwrite(self):
        report = {"status":"blocked"}
        dest = self.exp / "result.json"
        app.B.store_exclusive_json(dest, report)
        with self.assertRaises(FileExistsError):
            app.B.store_exclusive_json(dest, report)
        self.assertEqual(json.loads(dest.read_text())["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
