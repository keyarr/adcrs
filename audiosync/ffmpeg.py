import json
import shutil
import subprocess
from pathlib import Path


def require_tools():
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Missing required executable(s): " + ", ".join(missing))


def probe(path: str | Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def first_audio_stream(info: dict) -> dict:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    raise ValueError("No audio stream found")


def extract_audio(path: str | Path, sample_rate: int = 8000) -> tuple[bytes, int]:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-af", f"aresample={sample_rate}:async=1:first_pts=0",
            "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout, sample_rate
