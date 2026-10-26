"""CPU-only MLX 0.32.3 command-buffer lifecycle model.

Models Metal's actual capture rule: a command buffer is traced only if
created AFTER capture start and committed BEFORE capture stop. Not evidence
that real hardware/driver behavior matches this simplified model.
"""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "probes") not in sys.path:
    sys.path.insert(0, str(ROOT / "probes"))
SPEC = importlib.util.spec_from_file_location(
    "phase0a_capture_spike", ROOT / "tools/phase0a_capture_spike.py")
spike = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spike
SPEC.loader.exec_module(spike)


class FakeEncoder:
    def __init__(self, parent):
        self.parent = parent
        self.buffer = self.create_buffer()

    def create_buffer(self):
        self.parent.tick += 1
        return {"created": self.parent.tick, "committed": None, "dispatches": 0}

    def dispatch(self):
        self.buffer["dispatches"] += 1

    def commit(self):
        self.parent.tick += 1
        self.buffer["committed"] = self.parent.tick
        self.parent.buffers.append(self.buffer)
        # Mirrors MLX CommandEncoder::commit(), which immediately allocates
        # the next buffer BEFORE the next kernel or explicit synchronize.
        self.buffer = self.create_buffer()


class FakeMLX:
    gpu = "gpu"

    def __init__(self):
        self.tick = 0
        self.buffers = []
        self.default = FakeEncoder(self)
        self.current = self.default
        self.start = None
        self.stop = None
        self.metal = self

    def start_capture(self, path):
        self.tick += 1
        self.start = self.tick

    def stop_capture(self):
        self.tick += 1
        self.stop = self.tick

    def default_stream(self, device):
        return self.default

    def new_stream(self, device):
        return FakeEncoder(self)

    @contextmanager
    def stream(self, encoder):
        prev = self.current
        self.current = encoder
        try:
            yield
        finally:
            self.current = prev

    def eval(self, *outputs):
        self.current.commit()

    def synchronize(self, encoder):
        encoder.commit()  # commits even when no dispatch was encoded

    def dispatch(self):
        self.current.dispatch()
        return "lazy_array"

    def traced(self):
        return [b for b in self.buffers
                if self.start < b["created"] and b["committed"] < self.stop]


class MetalContractTests(unittest.TestCase):
    def warmed_default(self):
        mx = FakeMLX()
        mx.eval(mx.dispatch())  # validation run, leaves pre-created buffer
        mx.eval(mx.dispatch())  # warm-up, leaves *another* pre-created buffer
        return mx

    def test_preexisting_default_stream_is_empty_capture(self):
        mx = self.warmed_default()
        mx.start_capture("old.gputrace")
        mx.eval(mx.dispatch())
        mx.stop_capture()
        self.assertEqual(len(mx.traced()), 0)
        self.assertEqual(sum(x["dispatches"] for x in mx.traced()), 0)

    def test_just_synchronize_after_original_eval_is_not_a_fix(self):
        mx = self.warmed_default()
        mx.start_capture("old.gputrace")
        mx.eval(mx.dispatch())  # dispatch was in pre-capture buffer
        mx.synchronize(mx.default_stream(mx.gpu))  # commits fresh EMPTY buffer
        mx.stop_capture()
        self.assertEqual(len(mx.traced()), 1)
        self.assertEqual(sum(x["dispatches"] for x in mx.traced()), 0)

    def test_fresh_stream_has_one_buffer_one_dispatch(self):
        mx = self.warmed_default()
        spike.capture_once(mx, mx.dispatch, Path("fresh.gputrace"))
        self.assertEqual(len(mx.traced()), 1)
        self.assertEqual(sum(x["dispatches"] for x in mx.traced()), 1)

    def test_fallback_flushes_old_buffer_then_dispatches_new(self):
        mx = self.warmed_default()
        spike.capture_once(mx, mx.dispatch, Path("fallback.gputrace"),
                           mode="default-stream")
        self.assertEqual(len(mx.traced()), 1)
        self.assertEqual(sum(x["dispatches"] for x in mx.traced()), 1)

    def test_multiple_dispatches_fresh_stream_in_capture_window(self):
        from capture_lane import capture_batch
        mx = self.warmed_default()
        capture_batch(mx, mx.dispatch, Path("batch.gputrace"), 3,
                      "fresh-stream")
        self.assertGreaterEqual(len(mx.traced()), 1)
        self.assertEqual(sum(x["dispatches"] for x in mx.traced()), 3)

    def test_multiple_dispatches_default_preflush_in_capture_window(self):
        from capture_lane import capture_batch
        mx = self.warmed_default()
        capture_batch(mx, mx.dispatch, Path("batch.gputrace"), 3,
                      "default-stream")
        self.assertGreaterEqual(len(mx.traced()), 1)
        self.assertEqual(sum(x["dispatches"] for x in mx.traced()), 3)


if __name__ == "__main__":
    unittest.main()
