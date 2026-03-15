"""Pure geometry helpers for pose metrics (no deps beyond numpy)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def angle_three_points(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Angle at point b, in degrees (0–180)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


def horizontal_tilt_deg(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Degrees from horizontal of line p1 → p2 (0 = horizontal, 90 = vertical)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(math.degrees(math.atan2(abs(dy), max(abs(dx), 1e-9))))


def normalized_distance(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Euclidean distance in normalized image coordinates."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(math.sqrt(dx * dx + dy * dy))


# ---------------------------------------------------------------------------
# 3D geometry helpers (use MediaPipe world landmarks: metres, hip-centred)
# Coordinate convention: x=right, y=down, z=into-camera (depth)
# ---------------------------------------------------------------------------

def angle_three_points_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """Angle at point b in 3D space, in degrees (0–180)."""
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


def body_lean_angle_deg(
    shoulder_mid: tuple[float, float, float],
    ankle_mid: tuple[float, float, float],
) -> float:
    """Full-body lateral lean from vertical in degrees.

    Computes the angle between the ankle→shoulder vector and the vertical
    up direction.  0° = perfectly upright, increases with lateral lean.

    Uses world landmark coordinates where y-axis points *downward*, so the
    vertical-up reference is (0, -1, 0).
    """
    vec = np.array([
        shoulder_mid[0] - ankle_mid[0],
        shoulder_mid[1] - ankle_mid[1],
        shoulder_mid[2] - ankle_mid[2],
    ], dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 0.0
    vertical_up = np.array([0.0, -1.0, 0.0])
    cos_a = float(np.dot(vec / norm, vertical_up))
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(math.degrees(math.acos(cos_a)))


def edge_angle_proxy_deg(
    knee_mid: tuple[float, float, float],
    ankle_mid: tuple[float, float, float],
) -> float:
    """Lower-leg lateral inclination from vertical (edge-angle proxy).

    Measures how far the shin (knee→ankle segment) deviates from vertical.
    0° = vertical shin (no edge), 30-45° = aggressive carving lean.
    This is closer to the true ski-to-snow edge angle than the full-body lean.

    Uses world landmark coordinates where y-axis points *downward*.
    """
    vec = np.array([
        ankle_mid[0] - knee_mid[0],
        ankle_mid[1] - knee_mid[1],
        ankle_mid[2] - knee_mid[2],
    ], dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 0.0
    vertical_down = np.array([0.0, 1.0, 0.0])
    cos_a = float(np.dot(vec / norm, vertical_down))
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(math.degrees(math.acos(cos_a)))


def vertical_alignment_score(
    top: tuple[float, float],
    mid: tuple[float, float],
    bottom: tuple[float, float],
) -> float:
    """How well mid.x falls between top.x and bottom.x.

    Returns 0.0 (perfect vertical stack) to ~1.0 (max misalignment).
    Front-view sagittal-plane proxy for knee-over-ankle alignment.
    """
    span = abs(bottom[0] - top[0])
    if span < 1e-9:
        # top and bottom nearly identical x — measure how far mid deviates
        return min(1.0, abs(mid[0] - top[0]) * 10)
    expected_x = top[0] + (bottom[0] - top[0]) * 0.5
    deviation = abs(mid[0] - expected_x)
    return min(1.0, deviation / max(span, 1e-9))
