"""Safely validate gpudebug output without uploading traces or leaking paths."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "inspect_gpu_trace", ROOT / "tools/inspect_gpu_trace.py")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


class TraceInspectionTests(unittest.TestCase):
    def test_nonempty(self):
        result = probe.parse_summary(
            "Summary: 3 command buffers, 18 encoders, 412 draw calls")
        self.assertEqual(result["status"], "nonempty")
        self.assertEqual(result["command_buffers"], 3)

    def test_empty(self):
        self.assertEqual(
            probe.parse_summary("Summary: 0 command buffers, 0 encoders")["status"],
            "empty")
        self.assertEqual(
            probe.parse_summary("Command Buffers  : 2\nDispatch Calls: 0")["status"],
            "empty")

    def test_unknown_does_not_claim_success(self):
        self.assertEqual(probe.parse_summary("Could not read trace")["status"],
                         "unverified")

    def test_subprocess_errors_redacted(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "a.gputrace"
            trace.mkdir()
            with patch.object(probe.subprocess, "run", return_value=Mock(
                    returncode=1, stdout="", stderr="/Users/private/path")):
                result = probe.inspect(trace)
        self.assertEqual(result["status"], "unverified")
        self.assertNotIn("private", str(result))


if __name__ == "__main__":
    unittest.main()
