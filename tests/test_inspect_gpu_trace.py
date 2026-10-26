"""CPU tests: bundle size is diagnostic, never a capture-content verdict."""
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
    def test_empty_bundle_is_unverified_not_proven_empty(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "empty.gputrace"
            trace.mkdir()
            info = probe.inspect(trace)
        self.assertEqual(info["status"], "unverified")
        self.assertEqual(info["bundle_bytes"], 0)
        self.assertTrue(info["size_is_not_proof_of_commands"])

    def test_nonzero_bundle_is_unverified_not_proven_nonempty(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "nonempty.gputrace"
            trace.mkdir()
            (trace / "data").write_bytes(b"x" * 8192)
            info = probe.inspect(trace)
        self.assertEqual(info["bundle_bytes"], 8192)
        self.assertEqual(info["status"], "unverified")

    def test_missing_trace_is_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            info = probe.inspect(Path(td) / "gone.gputrace")
        self.assertEqual(info["status"], "unverified")
        self.assertNotIn("bundle_bytes", info)

    def test_unvalidated_gpudebug_output_cannot_claim_nonempty(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "a.gputrace"
            trace.mkdir()
            with patch.object(probe.subprocess, "run", return_value=Mock(
                    returncode=0, stdout="Command Buffers: 123",
                    stderr="/Users/private/path")):
                result = probe.inspect(trace, gpudebug=True)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["gpudebug_probe"], "ran_output_unvalidated")
        self.assertNotIn("private", str(result))
        self.assertNotIn("123", str(result))

    def test_gpudebug_error_cannot_leak_paths(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "a.gputrace"
            trace.mkdir()
            with patch.object(probe.subprocess, "run", return_value=Mock(
                    returncode=1, stdout="", stderr="/Users/private/path")):
                result = probe.inspect(trace, gpudebug=True)
        self.assertEqual(result["gpudebug_probe"], "gpudebug_failed")
        self.assertEqual(result["status"], "unverified")
        self.assertNotIn("private", str(result))


if __name__ == "__main__":
    unittest.main()
