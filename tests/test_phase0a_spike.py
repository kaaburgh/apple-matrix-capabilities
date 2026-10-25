"""CPU-only guard tests for the manual Phase 0a GPU capture spike."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools.phase0a_capture_spike import (
    PreflightError, capture_once, validate_preflight,
)


class TestCaptureSpikePreflight(unittest.TestCase):
    def test_no_mac(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaisesRegex(PreflightError, "Apple Silicon"):
                validate_preflight("Linux", "x86_64", "1",
                                   Path(p) / "trace.gputrace")

    def test_capture_flag_required(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaisesRegex(PreflightError, "MTL_CAPTURE_ENABLED"):
                validate_preflight("Darwin", "arm64", None,
                                   Path(p) / "trace.gputrace")

    def test_refuse_existing_trace_directory(self):
        with tempfile.TemporaryDirectory() as p:
            path = Path(p) / "trace.gputrace"
            path.mkdir()
            with self.assertRaisesRegex(PreflightError, "already exists"):
                validate_preflight("Darwin", "arm64", "1", path)

    def test_refuse_wrong_suffix(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaisesRegex(PreflightError, "end in .gputrace"):
                validate_preflight("Darwin", "arm64", "1", Path(p) / "trace.json")

    def test_refuse_missing_parent(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaisesRegex(PreflightError, "parent directory"):
                validate_preflight("Darwin", "arm64", "1",
                                   Path(p) / "missing" / "trace.gputrace")

    def test_synthetic_preflight_is_not_hardware_test(self):
        with tempfile.TemporaryDirectory() as p:
            validate_preflight("Darwin", "arm64", "1",
                               Path(p) / "trace.gputrace")


class TestCaptureLifecycle(unittest.TestCase):
    def test_one_dispatch_then_stop(self):
        mx = Mock()
        dispatch = Mock(return_value="array")
        capture_once(mx, dispatch, Path("trace.gputrace"))
        mx.metal.start_capture.assert_called_once_with("trace.gputrace")
        dispatch.assert_called_once()
        mx.eval.assert_called_once_with("array")
        mx.metal.stop_capture.assert_called_once()

    def test_stop_if_evaluation_fails(self):
        mx = Mock()
        mx.eval.side_effect = RuntimeError("synthetic failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            capture_once(mx, Mock(return_value="array"), Path("trace.gputrace"))
        mx.metal.stop_capture.assert_called_once()

    def test_no_stop_if_start_fails(self):
        mx = Mock()
        mx.metal.start_capture.side_effect = RuntimeError("could not start")
        with self.assertRaisesRegex(RuntimeError, "could not start"):
            capture_once(mx, Mock(), Path("trace.gputrace"))
        mx.metal.stop_capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
