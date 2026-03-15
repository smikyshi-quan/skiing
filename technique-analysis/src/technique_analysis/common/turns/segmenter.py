"""Turn segmenter using lateral CoM shift zero-crossings.

Signal priority:
  1. com_shift_x (world-landmark lateral CoM shift, metres) — physically meaningful,
     adapted from Alpine-ski-analyzer TurnAnalyzer._detect_turns().
  2. hip_tilt (degrees) — 2D fallback when world landmarks are unavailable.
"""

from __future__ import annotations

import numpy as np

from technique_analysis.common.contracts.models import FrameMetrics, TurnSummary

_CONF_THRESHOLD = 0.4

# Hysteresis thresholds (avoids false zero-crossings on noise)
_HYSTERESIS_COM = 0.002   # metres   — com_shift_x signal
_HYSTERESIS_HIP = 0.5     # degrees  — hip_tilt fallback signal

# Amplitude filters (turns with smaller swing are micro-corrections, not full turns)
_MIN_AMPLITUDE_COM = 0.01  # metres
_MIN_AMPLITUDE_HIP = 1.0   # degrees

_MIN_DURATION_S = 0.8
_MAX_DURATION_S = 6.0


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result.append(float(np.mean(values[lo:hi])))
    return result


def _confidence_weighted_mean(
    values: list[float | None], weights: list[float]
) -> float | None:
    valid = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    if total_w < 1e-9:
        return None
    return sum(v * w for v, w in valid) / total_w


def _detect_zero_crossings(
    smoothed: list[float],
    hysteresis: float,
) -> list[int]:
    """Schmitt-trigger zero-crossing detector.

    Tracks whether the signal is in a confirmed +/- state (above/below the
    hysteresis band) and records each transition, so gradual crossings on
    smoothed signals are handled correctly.
    """
    crossings: list[int] = []
    state = 0  # 0=ambiguous, +1=confirmed positive, -1=confirmed negative
    for i, v in enumerate(smoothed):
        if v > hysteresis:
            if state == -1:
                crossings.append(i)   # negative → positive crossing
            state = 1
        elif v < -hysteresis:
            if state == 1:
                crossings.append(i)   # positive → negative crossing
            state = -1
        # values inside [-hysteresis, +hysteresis] don't change state
    return crossings


def segment_turns(
    metrics_list: list[FrameMetrics],
    poses: list | None = None,
    min_duration_s: float = _MIN_DURATION_S,
    smoothing_window: int = 15,
) -> list[TurnSummary]:
    """Segment turns from lateral CoM oscillation (or hip-tilt fallback).

    Algorithm:
      1. Build signal from com_shift_x (world metres) if ≥30 % of frames have it,
         otherwise fall back to hip_tilt (degrees).
      2. Smooth signal with a rolling mean.
      3. Detect zero-crossings with hysteresis to suppress noise.
      4. Filter segments by duration and amplitude.
      5. Aggregate per-turn metrics with confidence weighting.
    """
    if not metrics_list:
        return []

    timestamps = [m.timestamp_s for m in metrics_list]

    # ---- Choose signal -------------------------------------------------------
    com_available = [m.com_shift_x for m in metrics_list if m.com_shift_x is not None]
    use_com = len(com_available) >= len(metrics_list) * 0.30

    if use_com:
        raw_signal = [
            m.com_shift_x if m.com_shift_x is not None else 0.0
            for m in metrics_list
        ]
        hysteresis = _HYSTERESIS_COM
        min_amplitude = _MIN_AMPLITUDE_COM
    else:
        raw_signal = [
            m.hip_tilt if m.hip_tilt is not None else 0.0
            for m in metrics_list
        ]
        hysteresis = _HYSTERESIS_HIP
        min_amplitude = _MIN_AMPLITUDE_HIP

    # Mask low-confidence frames to 0 so they don't pollute the signal
    masked = [
        v if m.pose_confidence > _CONF_THRESHOLD else 0.0
        for v, m in zip(raw_signal, metrics_list)
    ]

    # ---- Smooth & detect zero-crossings -------------------------------------
    smoothed = _rolling_mean(masked, smoothing_window)
    crossings = _detect_zero_crossings(smoothed, hysteresis)

    # Add sentinels so the first and last segments are considered
    boundaries = [0] + crossings + [len(metrics_list) - 1]
    boundaries = sorted(set(boundaries))

    # ---- Build turn segments ------------------------------------------------
    turns: list[TurnSummary] = []
    turn_idx = 0

    for seg in range(len(boundaries) - 1):
        start_i = boundaries[seg]
        end_i = boundaries[seg + 1]
        seg_metrics = metrics_list[start_i: end_i + 1]
        if not seg_metrics:
            continue

        start_s = seg_metrics[0].timestamp_s
        end_s = seg_metrics[-1].timestamp_s
        duration_s = end_s - start_s

        if duration_s < min_duration_s or duration_s > _MAX_DURATION_S:
            continue

        # Amplitude filter — ignore tiny oscillations
        seg_signal = smoothed[start_i: end_i + 1]
        amplitude = float(max(abs(v) for v in seg_signal)) if seg_signal else 0.0
        if amplitude < min_amplitude:
            continue

        # Turn direction: mean signal > 0 → right lean, < 0 → left lean
        mean_signal = float(np.mean(seg_signal))
        side = "right" if mean_signal > 0 else "left"

        # Confidence-weighted aggregation (only high-conf frames contribute)
        high_conf = [m for m in seg_metrics if m.pose_confidence >= _CONF_THRESHOLD]
        n_used = len(high_conf)
        weights = [m.pose_confidence for m in high_conf]
        avg_conf = float(np.mean(weights)) if weights else 0.0

        turns.append(TurnSummary(
            turn_idx=turn_idx,
            side=side,
            start_s=start_s,
            end_s=end_s,
            duration_s=duration_s,
            n_frames_used=n_used,
            avg_pose_confidence=avg_conf,
            avg_knee_flexion_L=_confidence_weighted_mean(
                [m.knee_flexion_L for m in high_conf], weights),
            avg_knee_flexion_R=_confidence_weighted_mean(
                [m.knee_flexion_R for m in high_conf], weights),
            avg_knee_flexion_diff=_confidence_weighted_mean(
                [m.knee_flexion_diff for m in high_conf], weights),
            avg_stance_width_ratio=_confidence_weighted_mean(
                [m.stance_width_ratio for m in high_conf], weights),
            avg_upper_body_quietness=_confidence_weighted_mean(
                [m.upper_body_quietness for m in high_conf], weights),
            avg_lean_angle=_confidence_weighted_mean(
                [m.lean_angle_deg for m in high_conf], weights),
            avg_edge_angle=_confidence_weighted_mean(
                [m.edge_angle_deg for m in high_conf], weights),
            avg_com_shift_3d=_confidence_weighted_mean(
                [m.com_shift_3d for m in high_conf], weights),
        ))
        turn_idx += 1

    return turns
