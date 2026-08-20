import csv
import json
from pathlib import Path

import numpy as np

from .ffmpeg import extract_audio


class SyncError(RuntimeError):
    pass


def _features(audio: np.ndarray, sample_rate: int, frame_ms: int = 80) -> np.ndarray:
    size = max(256, int(sample_rate * frame_ms / 1000))
    hop = size // 2
    if len(audio) < size:
        raise SyncError("Audio is shorter than the analysis frame")
    frames = np.lib.stride_tricks.sliding_window_view(audio, size)[::hop]
    window = np.hanning(size)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    spectrum = spectrum[:, 1:]
    edges = np.linspace(0, spectrum.shape[1], 17, dtype=int)
    bands = np.stack([spectrum[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(16)], axis=1)
    bands = np.log1p(bands)
    flux = np.maximum(0, np.diff(bands, axis=0, prepend=bands[:1])).mean(axis=1)
    result = np.column_stack((bands, flux))
    result -= result.mean(axis=0)
    scale = result.std(axis=0)
    result /= np.where(scale > 1e-8, scale, 1)
    return result.astype(np.float32)


def _match(template: np.ndarray, source: np.ndarray) -> tuple[int, float]:
    length, channels = template.shape
    if len(source) < length:
        raise SyncError("Source audio is shorter than the target analysis window")
    score = np.zeros(len(source) - length + 1, dtype=np.float64)
    for channel in range(channels):
        correlation = np.correlate(source[:, channel], template[:, channel], mode="valid")
        sums = np.convolve(source[:, channel] ** 2, np.ones(length), mode="valid")
        denom = np.sqrt(np.maximum(sums * np.sum(template[:, channel] ** 2), 1e-12))
        score += correlation / denom
    score /= channels
    index = int(np.argmax(score))
    return index, float(score[index])


def analyze(target: np.ndarray, source: np.ndarray, sample_rate: int, window_seconds: float = 30.0, step_seconds: float = 180.0) -> dict:
    if sample_rate <= 0:
        raise SyncError("Sample rate must be positive")
    if not np.isfinite(window_seconds) or not np.isfinite(step_seconds) or window_seconds <= 0 or step_seconds <= 0:
        raise SyncError("Window and step must be finite and positive")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(source)):
        raise SyncError("Audio contains non-finite samples")
    target_features = _features(target, sample_rate)
    source_features = _features(source, sample_rate)
    feature_hop_seconds = (max(256, int(sample_rate * 80 / 1000)) // 2) / sample_rate
    frame_step = feature_hop_seconds
    window = max(8, round(window_seconds / feature_hop_seconds))
    step = max(window, round(step_seconds / feature_hop_seconds))
    if len(target_features) < window:
        window = len(target_features)
    matches = []
    for start in range(0, len(target_features) - window + 1, step):
        source_start, confidence = _match(target_features[start:start + window], source_features)
        matches.append({"target_time": start * frame_step, "source_time": source_start * frame_step, "confidence": confidence})
    valid = [m for m in matches if m["confidence"] >= 0.25]
    if not valid:
        raise SyncError("No reliable audio correspondence found")
    target_times = np.array([m["target_time"] for m in valid])
    source_times = np.array([m["source_time"] for m in valid])
    coefficients = np.polyfit(target_times, source_times, 1) if len(valid) >= 2 else (1.0, source_times[0])
    ratio, offset = float(coefficients[0]), float(coefficients[1])
    residuals = source_times - (offset + ratio * target_times)
    robust = np.abs(residuals) <= max(0.35, float(np.median(np.abs(residuals)) * 3 + 0.05))
    confidence = float(np.clip(np.mean([m["confidence"] for m in valid]) * np.mean(robust), 0, 1))
    return {"offset": offset, "tempo_ratio": ratio, "confidence": confidence, "matches": matches, "segments": [{"target_start": 0.0, "target_end": len(target) / sample_rate, "source_start": offset, "tempo": ratio}]}


def analyze_files(target_path: str | Path, source_path: str | Path) -> dict:
    target_raw, rate = extract_audio(target_path)
    source_raw, source_rate = extract_audio(source_path, rate)
    if source_rate != rate:
        raise SyncError("Analysis sample rates differ")
    return analyze(np.frombuffer(target_raw, dtype=np.float32), np.frombuffer(source_raw, dtype=np.float32), rate)


def write_report(report: dict, path: str | Path, csv_path: str | Path | None = None) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n")
    if csv_path:
        with Path(csv_path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("target_time", "source_time", "confidence"))
            writer.writeheader()
            writer.writerows(report["matches"])
