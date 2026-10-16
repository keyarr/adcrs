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


def _match(template: np.ndarray, source: np.ndarray, exclude_frames: int = 25) -> tuple[int, float, float]:
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
    best = float(score[index])
    lo = max(0, index - exclude_frames)
    hi = min(len(score), index + exclude_frames + 1)
    masked = score.copy()
    masked[lo:hi] = -np.inf
    second = float(np.max(masked))
    return index, best, best - second


CONF_MIN = 0.25
MARGIN_MIN = 0.015
CLUSTER_TOL = 0.5
MIN_CONFIDENCE = 0.35


def _fit_level(points: list[tuple[float, float]]) -> tuple[float, float]:
    deltas = [s - t for t, s in points]
    if len(points) < 2:
        return 1.0, float(deltas[0])
    target = np.array([t for t, _ in points])
    source = np.array([s for _, s in points])
    slope, intercept = np.polyfit(target, source, 1)
    if abs(float(slope) - 1.0) > 0.1:
        return 1.0, float(np.median(deltas))
    return float(slope), float(intercept)


def analyze(target: np.ndarray, source: np.ndarray, sample_rate: int, window_seconds: float = 30.0, step_seconds: float = 60.0) -> dict:
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
    exclude = max(1, round(1.0 / feature_hop_seconds))
    for start in range(0, len(target_features) - window + 1, step):
        source_start, confidence, margin = _match(target_features[start:start + window], source_features, exclude)
        matches.append({"target_time": start * frame_step, "source_time": source_start * frame_step, "confidence": confidence, "margin": margin})
    reliable = [m for m in matches if m["confidence"] >= CONF_MIN and m["margin"] >= MARGIN_MIN]
    if len(reliable) < 2:
        raise SyncError("No reliable audio correspondence found")
    order = sorted(range(len(reliable)), key=lambda i: reliable[i]["source_time"] - reliable[i]["target_time"])
    groups: list[list[int]] = []
    for idx in order:
        delta = reliable[idx]["source_time"] - reliable[idx]["target_time"]
        if groups:
            prev = reliable[groups[-1][-1]]["source_time"] - reliable[groups[-1][-1]]["target_time"]
            if abs(delta - prev) <= CLUSTER_TOL:
                groups[-1].append(idx)
                continue
        groups.append([idx])
    edges = {matches[0]["target_time"], matches[-1]["target_time"]}
    stables = [
        g for g in groups
        if len(g) >= 2 or (reliable[g[0]]["target_time"] in edges and reliable[g[0]]["margin"] >= 0.1)
    ]
    if not any(len(g) >= 2 for g in stables):
        raise SyncError("No reliable audio correspondence found")
    main = max(stables, key=lambda g: (len(g), -min(reliable[i]["target_time"] for i in g)))
    main_points = [(reliable[i]["target_time"], reliable[i]["source_time"]) for i in main]
    ratio, offset = _fit_level(main_points)
    mean_conf = float(np.mean([reliable[i]["confidence"] for i in main]))
    step_sec = step * frame_step
    span = (max(t for t, _ in main_points) - min(t for t, _ in main_points) + step_sec) / (len(matches) * step_sec)
    confidence = float(np.clip(mean_conf * (len(main) / len(reliable)) * span, 0, 1))
    cuts = len(stables) - 1
    if cuts > 0:
        confidence = min(confidence * 0.5, 0.34)
    duration = len(target) / sample_rate
    if cuts == 0:
        segments = [{"target_start": 0.0, "target_end": duration, "source_start": offset, "tempo": ratio}]
    else:
        stables.sort(key=lambda g: min(reliable[i]["target_time"] for i in g))
        bounds = [0.0]
        for first, second in zip(stables[:-1], stables[1:]):
            prev_end = max(reliable[i]["target_time"] for i in first)
            next_start = min(reliable[i]["target_time"] for i in second)
            bounds.append((prev_end + next_start) / 2)
        bounds.append(duration)
        segments = []
        for group, start, end in zip(stables, bounds[:-1], bounds[1:]):
            points = [(reliable[i]["target_time"], reliable[i]["source_time"]) for i in group]
            tempo, start_offset = _fit_level(points)
            segments.append({"target_start": start, "target_end": end, "source_start": start_offset, "tempo": tempo})
    return {"offset": offset, "tempo_ratio": ratio, "confidence": confidence, "cuts": cuts, "matches": matches, "segments": segments}


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
            writer = csv.DictWriter(handle, fieldnames=("target_time", "source_time", "confidence", "margin"))
            writer.writeheader()
            for row in report["matches"]:
                writer.writerow({key: row.get(key, "") for key in ("target_time", "source_time", "confidence", "margin")})
