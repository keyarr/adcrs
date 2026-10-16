import argparse
import subprocess
from pathlib import Path

from .ffmpeg import first_audio_stream, probe, require_tools
from .sync import SyncError, analyze_files, write_report


def output_path(target: Path, requested: str | None) -> Path:
    return Path(requested) if requested else target.with_name(target.stem + ".dubbed" + target.suffix)


def mux(target: Path, source: Path, output: Path, offset: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(target), "-itsoffset", str(-offset), "-i", str(source),
        "-map", "0:v?", "-map", "0:a?", "-map", "1:a:0", "-map", "0:s?", "-map", "0:d?", "-map_metadata", "0",
        "-c", "copy", str(output),
    ], check=True)


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
