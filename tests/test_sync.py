import unittest

import numpy as np

from audiosync.sync import SyncError, analyze


def signal(length=24000, seed=4):
    rng = np.random.default_rng(seed)
    value = rng.normal(size=length).astype(np.float32)
    value[::97] += 4
    return value


class SyncTests(unittest.TestCase):
    def test_fixed_offset(self):
        target = signal()
        source = np.concatenate((np.zeros(1800, dtype=np.float32), target, np.zeros(1000, dtype=np.float32)))
        report = analyze(target, source, 800, window_seconds=6, step_seconds=6)
        self.assertAlmostEqual(report["offset"], 2.16, delta=0.2)

    def test_low_confidence_is_reported(self):
        target = np.zeros(24000, dtype=np.float32)
        source = np.zeros(26000, dtype=np.float32)
        with self.assertRaises(SyncError):
            analyze(target, source, 800, window_seconds=6, step_seconds=6)

    def test_invalid_parameters_are_rejected(self):
        audio = signal()
        for sample_rate in (0, -8000):
            with self.assertRaises(SyncError):
                analyze(audio, audio, sample_rate)
        for kwargs in (
            {"window_seconds": 0},
            {"window_seconds": -1},
            {"step_seconds": 0},
            {"step_seconds": -1},
            {"window_seconds": float("nan")},
            {"step_seconds": float("inf")},
        ):
            with self.assertRaises(SyncError):
                analyze(audio, audio, 800, **kwargs)

    def test_nonfinite_audio_is_rejected(self):
        audio = signal()
        audio[100] = np.nan
        with self.assertRaises(SyncError):
            analyze(audio, signal(), 800)

    def test_offset_with_lead_silence(self):
        sig = signal(length=16000)
        target = np.concatenate((np.zeros(4000, dtype=np.float32), sig))
        source = np.concatenate((np.zeros(7904, dtype=np.float32), sig))
        report = analyze(target, source, 800, window_seconds=6, step_seconds=6)
        self.assertAlmostEqual(report["offset"], 4.88, delta=0.25)

    def test_mux_preserves_audio_streams(self):
        from unittest.mock import patch
        from pathlib import Path
        from audiosync.cli import mux

        with patch("subprocess.run") as mock_run:
            mux(Path("target.mkv"), Path("source.mkv"), Path("out.mkv"), 6.1)
            args = mock_run.call_args[0][0]
            self.assertIn("0:a?", args)
            self.assertIn("1:a:0", args)
            self.assertNotIn("-shortest", args)
            self.assertEqual(args[args.index("-c") + 1], "copy")


if __name__ == "__main__":
    unittest.main()
