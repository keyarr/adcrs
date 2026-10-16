import unittest

import numpy as np

from audiosync.cli import pair_batch
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

        fake_info = {
            "streams": [
                {"codec_type": "video", "start_time": "6.084000"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "1448.108000"},
        }
        with patch("audiosync.cli.probe", return_value=fake_info):
            with patch("audiosync.cli._verify_dubbed"):
                with patch("subprocess.run") as mock_run:
                    mux(Path("target.mkv"), Path("source.mkv"), Path("out.mkv"), 6.084)
                    args = mock_run.call_args[0][0]
                    self.assertIn("0:a?", args)
                    self.assertIn("1:a:0", args)
                    self.assertNotIn("-itsoffset", args)
                    self.assertNotIn("+genpts", args)
                    self.assertNotIn("make_zero", args)
                    # sparse PGS/data must come from a separate demuxer of the
                    # same target, or a 1000s subtitle gap stalls A/V
                    # interleaving and dumps the dubbed AAC in one cluster.
                    inputs = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "-i"]
                    self.assertEqual(inputs, ["target.mkv", "source.mkv", "target.mkv"])
                    self.assertIn("2:s?", args)
                    self.assertIn("2:d?", args)
                    self.assertNotIn("0:s?", args)
                    self.assertNotIn("0:d?", args)
                    # dubbed track (second audio) is re-encoded, the rest is copied
                    dubbed = "1"
                    self.assertIn(f"-c:a:{dubbed}", args)
                    self.assertEqual(args[args.index(f"-c:a:{dubbed}") + 1], "aac")
                    filt = args[args.index(f"-filter:a:{dubbed}") + 1]
                    self.assertIn("atrim=start=6.084", filt)
                    self.assertIn("async=1", filt)
                    self.assertIn("whole_dur=", filt)
                    self.assertIn("language=por", args)


class PairTests(unittest.TestCase):
    def test_pairs_by_tag_regardless_of_order(self):
        targets = ["Show S02E03 HQ.mkv", "Show S02E01 HQ.mkv"]
        sources = ["dubbed S02E01.mkv", "dubbed S02E03.mkv"]
        self.assertEqual(pair_batch(targets, sources), [(targets[1], sources[0]), (targets[0], sources[1])])

    def test_single_without_tag_still_works(self):
        self.assertEqual(pair_batch(["hq.mkv"], ["dub.mkv"]), [("hq.mkv", "dub.mkv")])

    def test_single_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            pair_batch(["A S02E01.mkv"], ["B S02E02.mkv"])

    def test_batch_requires_tags_and_matching_sets(self):
        with self.assertRaises(ValueError):
            pair_batch(["A S02E01.mkv", "B.mkv"], ["C S02E01.mkv", "D S02E02.mkv"])
        with self.assertRaises(ValueError):
            pair_batch(["A S02E01.mkv", "B S02E01.mkv"], ["C S02E01.mkv", "D S02E02.mkv"])
        with self.assertRaises(ValueError):
            pair_batch(["A S02E01.mkv", "B S02E02.mkv"], ["C S02E01.mkv", "D S02E03.mkv"])
        with self.assertRaises(ValueError):
            pair_batch(["a", "b"], ["c"])


if __name__ == "__main__":
    unittest.main()
