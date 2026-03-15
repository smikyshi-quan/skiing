"""Video I/O utilities for technique-analysis."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from technique_analysis.common.contracts.models import VideoMetadata


def recommend_config(metadata: "VideoMetadata") -> tuple[float | None, int]:
    """Return (max_fps, max_dimension) tuned to the video's resolution and native FPS.

    Resolution tiers:
      4K  (≥3840 long side) → max_dimension=1920, max_fps=20
      2K  (≥2560)           → max_dimension=1440, max_fps=20
      1080p (≥1920)         → max_dimension=1080, max_fps=15
      720p or lower         → no resize,           max_fps=15 (or native if ≤15)

    FPS: never sub-sample below 15 fps; if native FPS ≤ 15 keep all frames.
    """
    long_side = max(metadata.width, metadata.height)

    if long_side >= 3840:
        max_dimension = 1920
    elif long_side >= 2560:
        max_dimension = 1440
    elif long_side >= 1920:
        max_dimension = 1080
    else:
        max_dimension = long_side  # no resize

    if metadata.fps <= 15.0:
        max_fps = None  # keep all frames
    else:
        max_fps = 20.0  # sample at 20 fps regardless of native rate

    return max_fps, max_dimension


def probe_video(video_path: Path) -> VideoMetadata:
    """Read video metadata without decoding frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        width=width,
        height=height,
        duration_s=duration_s,
        frame_count=frame_count,
    )


def _resize_if_needed(frame: np.ndarray, max_dimension: int | None) -> np.ndarray:
    if max_dimension is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_dimension:
        return frame
    scale = max_dimension / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def iter_frames(
    video_path: Path,
    max_fps: float | None = None,
    max_dimension: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, timestamp_s, bgr_frame) from video.

    Applies frame skipping for max_fps and optional resize.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = 1
    if max_fps is not None and max_fps > 0 and native_fps > max_fps:
        stride = max(1, int(round(native_fps / max_fps)))
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                timestamp_s = frame_idx / native_fps
                yield frame_idx, timestamp_s, _resize_if_needed(frame, max_dimension)
            frame_idx += 1
    finally:
        cap.release()
