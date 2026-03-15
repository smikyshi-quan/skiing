"""CSV writer for FrameMetrics."""

from __future__ import annotations

import csv
from pathlib import Path

from technique_analysis.common.contracts.models import FrameMetrics

_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "pose_confidence",
    "knee_flexion_L",
    "knee_flexion_R",
    "hip_angle_L",
    "hip_angle_R",
    "shoulder_tilt",
    "hip_tilt",
    "knee_flexion_diff",
    "hip_height_diff",
    "stance_width_ratio",
    "upper_body_quietness",
    "hip_knee_ankle_alignment_L",
    "hip_knee_ankle_alignment_R",
    # 3D metrics
    "lean_angle_deg",
    "edge_angle_deg",
    "com_shift_x",
    "com_shift_3d",
    "knee_angle_3d",
    # Composite scores
    "overall_score",
    "movement_quality",
]


def _fmt(v: float | None) -> float | str:
    return "" if v is None else v


def write_metrics_csv(metrics_list: list[FrameMetrics], output_path: Path) -> None:
    """Write all FrameMetrics to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({
                "frame_idx": m.frame_idx,
                "timestamp_s": m.timestamp_s,
                "pose_confidence": m.pose_confidence,
                "knee_flexion_L": _fmt(m.knee_flexion_L),
                "knee_flexion_R": _fmt(m.knee_flexion_R),
                "hip_angle_L": _fmt(m.hip_angle_L),
                "hip_angle_R": _fmt(m.hip_angle_R),
                "shoulder_tilt": _fmt(m.shoulder_tilt),
                "hip_tilt": _fmt(m.hip_tilt),
                "knee_flexion_diff": _fmt(m.knee_flexion_diff),
                "hip_height_diff": _fmt(m.hip_height_diff),
                "stance_width_ratio": _fmt(m.stance_width_ratio),
                "upper_body_quietness": _fmt(m.upper_body_quietness),
                "hip_knee_ankle_alignment_L": _fmt(m.hip_knee_ankle_alignment_L),
                "hip_knee_ankle_alignment_R": _fmt(m.hip_knee_ankle_alignment_R),
                "lean_angle_deg": _fmt(m.lean_angle_deg),
                "edge_angle_deg": _fmt(m.edge_angle_deg),
                "com_shift_x": _fmt(m.com_shift_x),
                "com_shift_3d": _fmt(m.com_shift_3d),
                "knee_angle_3d": _fmt(m.knee_angle_3d),
                "overall_score": _fmt(m.overall_score),
                "movement_quality": m.movement_quality or "",
            })
