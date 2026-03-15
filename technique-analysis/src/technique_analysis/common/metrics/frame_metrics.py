"""Per-frame metric computation from FramePose."""

from __future__ import annotations

import math

import numpy as np

from technique_analysis.common.contracts.models import FrameMetrics, FramePose
from technique_analysis.common.metrics.geometry import (
    angle_three_points,
    angle_three_points_3d,
    body_lean_angle_deg,
    edge_angle_proxy_deg,
    horizontal_tilt_deg,
    normalized_distance,
    vertical_alignment_score,
)

# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NOSE = 0

_MIN_VIS = 0.3  # per-landmark minimum visibility to use in computation


def _xy(pose: FramePose, idx: int) -> tuple[float, float] | None:
    """Return (x, y) of landmark idx if sufficiently visible, else None."""
    if idx >= len(pose.landmarks):
        return None
    lm = pose.landmarks[idx]
    if lm.visibility < _MIN_VIS:
        return None
    return (lm.x, lm.y)


def _xyz_world(pose: FramePose, idx: int) -> tuple[float, float, float] | None:
    """Return (x, y, z) world landmark if available and visible, else None."""
    if pose.world_landmarks is None:
        return None
    if idx >= len(pose.world_landmarks):
        return None
    lm = pose.world_landmarks[idx]
    if lm.visibility < _MIN_VIS:
        return None
    return (lm.x, lm.y, lm.z)


def _midpoint_3d(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def compute_frame_metrics(pose: FramePose, torso_size: float) -> FrameMetrics:
    """Compute all scalar metrics for one frame."""
    lm = lambda idx: _xy(pose, idx)  # noqa: E731
    wm = lambda idx: _xyz_world(pose, idx)  # noqa: E731

    # --- 2D metrics (existing) ---

    # Knee flexion (hip-knee-ankle angle)
    knee_flexion_L = None
    p_lhip, p_lknee, p_lankle = lm(L_HIP), lm(L_KNEE), lm(L_ANKLE)
    if p_lhip and p_lknee and p_lankle:
        knee_flexion_L = angle_three_points(p_lhip, p_lknee, p_lankle)

    knee_flexion_R = None
    p_rhip, p_rknee, p_rankle = lm(R_HIP), lm(R_KNEE), lm(R_ANKLE)
    if p_rhip and p_rknee and p_rankle:
        knee_flexion_R = angle_three_points(p_rhip, p_rknee, p_rankle)

    # Hip angle (shoulder-hip-knee)
    hip_angle_L = None
    p_lshoulder = lm(L_SHOULDER)
    if p_lshoulder and p_lhip and p_lknee:
        hip_angle_L = angle_three_points(p_lshoulder, p_lhip, p_lknee)

    hip_angle_R = None
    p_rshoulder = lm(R_SHOULDER)
    if p_rshoulder and p_rhip and p_rknee:
        hip_angle_R = angle_three_points(p_rshoulder, p_rhip, p_rknee)

    # Shoulder tilt
    shoulder_tilt = None
    if p_lshoulder and p_rshoulder:
        shoulder_tilt = horizontal_tilt_deg(p_lshoulder, p_rshoulder)

    # Hip tilt
    hip_tilt = None
    if p_lhip and p_rhip:
        hip_tilt = horizontal_tilt_deg(p_lhip, p_rhip)

    # Knee flexion diff
    knee_flexion_diff = None
    if knee_flexion_L is not None and knee_flexion_R is not None:
        knee_flexion_diff = abs(knee_flexion_L - knee_flexion_R)

    # Hip height diff
    hip_height_diff = None
    if p_lhip and p_rhip:
        hip_height_diff = abs(p_lhip[1] - p_rhip[1])

    # Stance width ratio: ankle-to-ankle / hip-to-hip distance
    stance_width_ratio = None
    if p_lankle and p_rankle and p_lhip and p_rhip:
        ankle_width = normalized_distance(p_lankle, p_rankle)
        hip_width = normalized_distance(p_lhip, p_rhip)
        if hip_width > 1e-6:
            stance_width_ratio = ankle_width / hip_width

    # Hip-knee-ankle vertical alignment
    hip_knee_ankle_alignment_L = None
    if p_lhip and p_lknee and p_lankle:
        hip_knee_ankle_alignment_L = vertical_alignment_score(p_lhip, p_lknee, p_lankle)

    hip_knee_ankle_alignment_R = None
    if p_rhip and p_rknee and p_rankle:
        hip_knee_ankle_alignment_R = vertical_alignment_score(p_rhip, p_rknee, p_rankle)

    # --- 3D metrics (world landmarks) ---

    w_l_shoulder = wm(L_SHOULDER)
    w_r_shoulder = wm(R_SHOULDER)
    w_l_hip = wm(L_HIP)
    w_r_hip = wm(R_HIP)
    w_l_knee = wm(L_KNEE)
    w_r_knee = wm(R_KNEE)
    w_l_ankle = wm(L_ANKLE)
    w_r_ankle = wm(R_ANKLE)

    shoulder_mid_w = _midpoint_3d(w_l_shoulder, w_r_shoulder)
    hip_mid_w = _midpoint_3d(w_l_hip, w_r_hip)
    knee_mid_w = _midpoint_3d(w_l_knee, w_r_knee)
    ankle_mid_w = _midpoint_3d(w_l_ankle, w_r_ankle)

    # Full-body lateral lean from vertical
    # Sanity: >80° means shoulder is below ankle level — bad detection, discard.
    lean_angle_deg = None
    if shoulder_mid_w is not None and ankle_mid_w is not None:
        val = body_lean_angle_deg(shoulder_mid_w, ankle_mid_w)
        lean_angle_deg = val if val <= 80.0 else None

    # Lower-leg edge-angle proxy (knee→ankle inclination)
    # Sanity: >75° is beyond max realistic carving angle, discard.
    edge_angle_deg = None
    if knee_mid_w is not None and ankle_mid_w is not None:
        val = edge_angle_proxy_deg(knee_mid_w, ankle_mid_w)
        edge_angle_deg = val if val <= 75.0 else None

    # CoM shift: ankle midpoint position in world x (lateral) and x-z plane
    # World coords are hip-centred so hip ≈ (0,0,0); ankle_x reflects lateral imbalance
    com_shift_x = None
    com_shift_3d = None
    if ankle_mid_w is not None:
        com_shift_x = float(ankle_mid_w[0])
        com_shift_3d = float(math.sqrt(ankle_mid_w[0] ** 2 + ankle_mid_w[2] ** 2))

    # 3D average knee angle using world landmarks
    knee_angle_3d = None
    ka3d_vals = []
    if w_l_hip and w_l_knee and w_l_ankle:
        ka3d_vals.append(angle_three_points_3d(w_l_hip, w_l_knee, w_l_ankle))
    if w_r_hip and w_r_knee and w_r_ankle:
        ka3d_vals.append(angle_three_points_3d(w_r_hip, w_r_knee, w_r_ankle))
    if ka3d_vals:
        knee_angle_3d = float(np.mean(ka3d_vals))

    return FrameMetrics(
        frame_idx=pose.frame_idx,
        timestamp_s=pose.timestamp_s,
        pose_confidence=pose.pose_confidence,
        knee_flexion_L=knee_flexion_L,
        knee_flexion_R=knee_flexion_R,
        hip_angle_L=hip_angle_L,
        hip_angle_R=hip_angle_R,
        shoulder_tilt=shoulder_tilt,
        hip_tilt=hip_tilt,
        knee_flexion_diff=knee_flexion_diff,
        hip_height_diff=hip_height_diff,
        stance_width_ratio=stance_width_ratio,
        upper_body_quietness=None,  # filled in second pass
        hip_knee_ankle_alignment_L=hip_knee_ankle_alignment_L,
        hip_knee_ankle_alignment_R=hip_knee_ankle_alignment_R,
        lean_angle_deg=lean_angle_deg,
        edge_angle_deg=edge_angle_deg,
        com_shift_x=com_shift_x,
        com_shift_3d=com_shift_3d,
        knee_angle_3d=knee_angle_3d,
    )


def compute_upper_body_quietness(
    metrics_list: list[FrameMetrics],
    poses: list[FramePose | None],
    window: int = 30,
) -> list[FrameMetrics]:
    """Second pass: fill upper_body_quietness via rolling variance of nose x position."""
    if not metrics_list:
        return metrics_list

    # Collect nose x per frame
    nose_x: list[float | None] = []
    for p in poses:
        if p is not None and len(p.landmarks) > NOSE:
            lm = p.landmarks[NOSE]
            nose_x.append(lm.x if lm.visibility >= _MIN_VIS else None)
        else:
            nose_x.append(None)

    half = window // 2
    result: list[FrameMetrics] = []
    for i, m in enumerate(metrics_list):
        lo = max(0, i - half)
        hi = min(len(nose_x), i + half + 1)
        window_vals = [v for v in nose_x[lo:hi] if v is not None]
        quietness = float(np.var(window_vals)) if len(window_vals) >= 3 else None
        result.append(FrameMetrics(
            frame_idx=m.frame_idx,
            timestamp_s=m.timestamp_s,
            pose_confidence=m.pose_confidence,
            knee_flexion_L=m.knee_flexion_L,
            knee_flexion_R=m.knee_flexion_R,
            hip_angle_L=m.hip_angle_L,
            hip_angle_R=m.hip_angle_R,
            shoulder_tilt=m.shoulder_tilt,
            hip_tilt=m.hip_tilt,
            knee_flexion_diff=m.knee_flexion_diff,
            hip_height_diff=m.hip_height_diff,
            stance_width_ratio=m.stance_width_ratio,
            upper_body_quietness=quietness,
            hip_knee_ankle_alignment_L=m.hip_knee_ankle_alignment_L,
            hip_knee_ankle_alignment_R=m.hip_knee_ankle_alignment_R,
            lean_angle_deg=m.lean_angle_deg,
            edge_angle_deg=m.edge_angle_deg,
            com_shift_x=m.com_shift_x,
            com_shift_3d=m.com_shift_3d,
            knee_angle_3d=m.knee_angle_3d,
        ))
    return result
