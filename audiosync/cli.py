import argparse
import subprocess
from pathlib import Path

from .ffmpeg import first_audio_stream, probe, require_tools
from .sync import SyncError, analyze_files, write_report


def output_path(target: Path, requested: str | None) -> Path:
    return Path(requested) if requested else target.with_name(target.stem + ".dubbed" + target.suffix)


def _target_layout(info: dict) -> tuple[int, float | None]:
    streams = info.get("streams", [])
    audio_count = sum(1 for stream in streams if stream.get("codec_type") == "audio")
    duration: float | None = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    start = 0.0
    for stream in streams:
        if stream.get("codec_type") == "video":
            try:
                start = float(stream.get("start_time", 0.0))
            except (TypeError, ValueError):
                start = 0.0
            break
    if duration is not None and duration > start:
        return audio_count, duration - start
    return audio_count, None


def _dub_filter(offset: float, length: float | None) -> str:
    if offset > 0.001:
        head = f"atrim=start={offset:.3f},asetpts=PTS-STARTPTS"
    elif offset < -0.001:
        delay_ms = int(round(-offset * 1000))
        head = f"adelay=delays={delay_ms}:all=1,asetpts=PTS-STARTPTS"
    else:
        head = "asetpts=PTS-STARTPTS"
    # Resample so the new track matches the target base, pad short sources
    # with silence and cut long ones at the target length. whole_dur bounds
    # apad so a short source can't produce an infinite stream when length
    # is unknown; async fills edit-list gaps instead of drifting.
    tail = ",aresample=48000:async=1"
    if length is not None and length > 0:
        tail += f",apad=whole_dur={length:.3f},atrim=0:{length:.3f}"
    else:
        tail += ",apad"
    return head + tail


def _verify_dubbed(output: Path, length: float | None) -> None:
    info = probe(output)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("Mux produced no audio streams")
    dubbed = streams[-1]
    dubbed_idx = len(streams) - 1
    try:
        rate = int(dubbed.get("sample_rate") or 0)
    except (TypeError, ValueError):
        rate = 0
    if rate <= 0:
        raise ValueError(f"Dubbed track has invalid sample rate: {dubbed.get('sample_rate')!r}")
    if length is not None and length > 5:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-map", f"0:a:{dubbed_idx}", "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1"],
            check=True,
            capture_output=True,
        )
        import numpy as np

        samples = np.frombuffer(result.stdout, dtype=np.float32)
        total = len(samples) / 8000 if len(samples) else 0
        if total < length - 2.0:
            raise ValueError(f"Dubbed audio too short: {total:.1f}s, expected ~{length:.1f}s")
        # Leading/trailing silence is normal (fade-in, padding for short
        # sources), so only the middle must carry signal. A fully silent
        # middle means the trim/delay ate the content.
        mid = len(samples) // 2
        window = samples[mid:mid + 8000]
        if len(window) > 100 and float(np.max(np.abs(window))) < 0.0005:
            raise ValueError(f"Dubbed track is silent in the middle ({mid / 8000:.1f}s)")
        if float(np.max(np.abs(samples))) < 0.01:
            raise ValueError("Dubbed track has no audible content")


def mux(target: Path, source: Path, output: Path, offset: float) -> None:
    target_info = probe(target)
    audio_count, length = _target_layout(target_info)
    dubbed = audio_count  # output audio index of 1:a:0
    # No +genpts / make_zero: those rewrite HEVC/TrueHD/PGS timestamps on
    # -c copy streams and VLC (strict Cues + faad) aborts while ffmpeg
    # tolerates it. Keep original timestamps, bake the offset into the
    # re-encoded dubbed track only.
    # Sparse PGS/data (e.g. a 1000s gap between Signs/Songs events) shares
    # input 0 with video/audio. ffmpeg's interleave scheduler stalls the
    # streams that run ahead of the most delayed one, so the whole dubbed
    # AAC got dumped in one cluster ~1GB away from its video. Demuxing
    # s/d from a separate instance of the same target file decouples the
    # gap from A/V scheduling without -max_interleave_delta 0 (2.6GB RAM).
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(target), "-i", str(source), "-i", str(target),
        "-map", "0:v?", "-map", "0:a?", "-map", "1:a:0", "-map", "2:s?", "-map", "2:d?", "-map_metadata", "0",
        "-c:v", "copy", "-c:s", "copy", "-c:d", "copy", "-c:a", "copy",
        f"-c:a:{dubbed}", "aac", f"-filter:a:{dubbed}", _dub_filter(offset, length),
        f"-b:a:{dubbed}", "192k", f"-ar:a:{dubbed}", "48000", f"-ac:a:{dubbed}", "2",
        f"-metadata:s:a:{dubbed}", "language=por",
        "-disposition:a", "0", f"-disposition:a:{dubbed}", "default",
        "-max_muxing_queue_size", "9999",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    _verify_dubbed(output, length)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize dubbed audio to a target video")
    parser.add_argument("target", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report")
    parser.add_argument("--csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        require_tools()
        target_info = probe(args.target)
        source_info = probe(args.source)
        first_audio_stream(target_info)
        first_audio_stream(source_info)
        print("Analyzing...")
        report = analyze_files(args.target, args.source)
        print(f"Initial offset: {report['offset']:+.3f} s")
        print(f"Tempo correction: {(report['tempo_ratio'] - 1) * 100:+.4f}%")
        print("Cuts detected: 0")
        print(f"Confidence: {report['confidence']:.0%}")
        if args.report or args.csv:
            write_report(report, args.report or "audiosync-report.json", args.csv)
        if report["confidence"] < 0.55:
            print("Result: MANUAL REVIEW")
            return 2
        if args.dry_run:
            print("Dry run: no output created")
            return 0
        destination = output_path(args.target, args.output)
        print("Muxing...")
        mux(args.target, args.source, destination, report["offset"])
        print(f"Output: {destination}")
        return 0
    except (OSError, ValueError, SyncError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
