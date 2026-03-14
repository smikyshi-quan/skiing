"""Overlay video renderer — teal skeleton, YOLO bbox, EDGE MAX primary metric.

Visual design matched to the reference standard:
  - Teal/cyan skeleton (#00FFCC) drawn with thick lines (4px at 1080p, scales up)
  - YOLO detection bounding box always visible (shows which person is tracked)
  - EDGE MAX as the primary HUD number, large and prominent
  - Arm/wrist connections drawn only when visibility > 0.5 (suppresses pole
    artefacts where MediaPipe places wrist landmarks on the ski pole tip)
  - Core body (torso + legs) drawn at visibility > 0.25 (always show the
    skeleton when the skier is detected, even in partial occlusion)
  - Joints: filled circles, colour-coded — bright teal on torso/hips/knees,
    softer yellow on feet, red on wrists (arms)
"""

from __future__ import annotations

import bisect
from pathlib import Path

import cv2
import numpy as np

from technique_analysis.common.contracts.models import (
    FrameMetrics,
    FramePose,
    PoseLandmark,
    TurnSummary,
)

# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------

# Core body connections (torso + legs) — drawn when vis > VIS_BODY
_CONNECTIONS_BODY: list[tuple[int, int]] = [
    (11, 12),           # shoulders
    (11, 23), (12, 24), # torso sides
    (23, 24),           # hips
    (23, 25), (25, 27), # L leg
    (24, 26), (26, 28), # R leg
    (27, 31), (28, 32), # L/R ankle→foot
]

# Arm connections — drawn only when vis > VIS_ARM (higher = pole filter)
_CONNECTIONS_ARMS: list[tuple[int, int]] = [
    (11, 13), (13, 15), # L arm
    (12, 14), (14, 16), # R arm
]

# Head spine
_CONNECTIONS_HEAD: list[tuple[int, int]] = [
    (0, 11), (0, 12),   # nose → shoulders
]

_VIS_BODY = 0.25   # threshold for core body connections
_VIS_ARM  = 0.55   # higher threshold — suppresses pole artefacts
_VIS_HEAD = 0.35

# Key joint groups for rendering
_JOINTS_TORSO   = [0, 11, 12, 23, 24]      # nose, shoulders, hips
_JOINTS_LEGS    = [25, 26, 27, 28, 31, 32]  # knees, ankles, feet
_JOINTS_ARMS    = [13, 14, 15, 16]          # elbows, wrists

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_TEAL        = (204, 255, 0)    # #00FFCC in BGR → bright teal
_TEAL_LIGHT  = (230, 255, 153)  # lighter teal for limbs
_YELLOW      = (0, 220, 255)    # feet joints
_RED_WRIST   = (60, 60, 255)    # wrist/elbow joints
_WHITE       = (255, 255, 255)
_BLACK       = (0, 0, 0)
_ORANGE_WARN = (0, 140, 255)    # low-confidence warning
_BBOX_COLOR  = (204, 255, 0)    # same teal as skeleton
_STALE_DIM   = 0.35             # dim factor when skeleton is held (not interpolated)

_REFERENCE_DIM = 1080


def _hud_scale(w: int, h: int) -> float:
    return max(w, h) / _REFERENCE_DIM


def _shadow(frame: np.ndarray, text: str, x: int, y: int,
            scale: float, thickness: int, color: tuple) -> None:
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, _BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_skeleton(
    frame: np.ndarray,
    pose: FramePose,
    w: int, h: int,
    dim: float = 1.0,
) -> None:
    lms = pose.landmarks

    def px(idx: int) -> tuple[int, int] | None:
        if idx >= len(lms):
            return None
        lm = lms[idx]
        return (int(lm.x * w), int(lm.y * h))

    def vis(idx: int) -> float:
        return lms[idx].visibility if idx < len(lms) else 0.0

    line_w = max(2, int(4 * dim + 0.5))
    joint_r = max(3, int(7 * dim + 0.5))
    arm_joint_r = max(2, int(5 * dim + 0.5))

    # --- Core body connections ---
    for a, b in _CONNECTIONS_BODY:
        if min(vis(a), vis(b)) < _VIS_BODY:
            continue
        pa, pb = px(a), px(b)
        if pa and pb:
            g = int(255 * min(vis(a), vis(b)) * dim)
            color = (max(0, 204 - (255 - g) // 2), min(255, 255 * dim), max(0, g * dim))
            cv2.line(frame, pa, pb, _TEAL if dim > 0.5 else _TEAL_LIGHT,
                     line_w, cv2.LINE_AA)

    # --- Arm connections (filtered — pole artefact suppression) ---
    for a, b in _CONNECTIONS_ARMS:
        if min(vis(a), vis(b)) < _VIS_ARM:
            continue
        pa, pb = px(a), px(b)
        if pa and pb:
            # Extra check: arm segment shouldn't be longer than the torso
            # (if so, the landmark is probably on a ski pole)
            if _arm_is_plausible(lms, a, b):
                cv2.line(frame, pa, pb, _TEAL_LIGHT, max(1, line_w - 1), cv2.LINE_AA)

    # --- Head connections ---
    for a, b in _CONNECTIONS_HEAD:
        if min(vis(a), vis(b)) < _VIS_HEAD:
            continue
        pa, pb = px(a), px(b)
        if pa and pb:
            cv2.line(frame, pa, pb, _TEAL_LIGHT, max(1, line_w - 1), cv2.LINE_AA)

    # --- Key joints ---
    for idx in _JOINTS_TORSO:
        p = px(idx)
        if p and vis(idx) >= _VIS_BODY:
            cv2.circle(frame, p, joint_r, _TEAL, -1, cv2.LINE_AA)
            cv2.circle(frame, p, joint_r + 1, _BLACK, 1, cv2.LINE_AA)

    for idx in _JOINTS_LEGS:
        p = px(idx)
        if p and vis(idx) >= _VIS_BODY:
            c = _YELLOW if idx in (31, 32) else _TEAL
            cv2.circle(frame, p, joint_r, c, -1, cv2.LINE_AA)

    for idx in _JOINTS_ARMS:
        p = px(idx)
        if p and vis(idx) >= _VIS_ARM:
            cv2.circle(frame, p, arm_joint_r, _RED_WRIST, -1, cv2.LINE_AA)


def _arm_is_plausible(lms: list[PoseLandmark], a: int, b: int) -> bool:
    """Return False if the arm segment is implausibly long (pole artefact)."""
    if len(lms) < 29:
        return True
    # Torso height as reference (shoulder y to hip y)
    sh_y = (lms[11].y + lms[12].y) / 2
    hip_y = (lms[23].y + lms[24].y) / 2
    torso_h = abs(hip_y - sh_y)
    if torso_h < 1e-4:
        return True
    la, lb = lms[a], lms[b]
    seg_len = ((la.x - lb.x) ** 2 + (la.y - lb.y) ** 2) ** 0.5
    # If segment > 0.9× torso height, likely pole artefact
    return seg_len < torso_h * 0.9


def _draw_bbox(
    frame: np.ndarray,
    pose: FramePose,
    scale: float,
) -> None:
    """Draw the YOLO detection bounding box.

    Suppressed when:
    - No detection bbox available
    - Bbox is very small (skier too far — tiny box is noise)
    - Tracking quality is low (gap-filled / stale)
    """
    if pose.detection_bbox is None:
        return
    # Don't show bbox for gap-filled/stale poses — avoids confusing
    # "frozen" box when skeleton has moved via interpolation
    if pose.tracking_quality < 0.6:
        return
    x1, y1, x2, y2 = pose.detection_bbox
    # Skip very small boxes (skier too distant — adds visual noise)
    if (x2 - x1) * (y2 - y1) < 5000:
        return
    thickness = max(2, int(3 * scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BBOX_COLOR, thickness, cv2.LINE_AA)
    # Small corner accents (like the reference app)
    corner = max(8, int(16 * scale))
    ct = thickness + 1
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), _WHITE, ct, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), _WHITE, ct, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    metrics: FrameMetrics | None,
    pose: FramePose | None,
    turn_idx: int | None,
    turn_side: str | None,
    scale: float,
    is_held: bool,
) -> None:
    if metrics is None:
        return

    h_frame, w_frame = frame.shape[:2]
    hud_x = int(12 * scale)
    y = int(30 * scale)
    line_h = int(26 * scale)

    conf = metrics.pose_confidence
    conf_color = (0, 200, 80) if conf >= 0.4 else _ORANGE_WARN

    # --- PRIMARY METRIC: EDGE MAX (large, like reference app) ---
    if metrics.edge_angle_deg is not None:
        edge_text = f"EDGE: {metrics.edge_angle_deg:.0f}deg"
        edge_scale = 1.4 * scale
        edge_thick = max(2, int(2.5 * scale))
        # Draw to upper-right corner (matches reference)
        (tw, th), _ = cv2.getTextSize(edge_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       edge_scale, edge_thick)
        ex = w_frame - tw - int(20 * scale)
        ey = int(50 * scale)
        # Background pill
        pad = int(8 * scale)
        cv2.rectangle(frame, (ex - pad, ey - th - pad),
                      (ex + tw + pad, ey + pad),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (ex - pad, ey - th - pad),
                      (ex + tw + pad, ey + pad),
                      _TEAL, max(1, int(2 * scale)))
        _shadow(frame, edge_text, ex, ey, edge_scale, edge_thick, _TEAL)
    elif metrics.lean_angle_deg is not None:
        # Fallback if edge not available
        lean_text = f"LEAN: {metrics.lean_angle_deg:.0f}deg"
        _shadow(frame, lean_text, w_frame - int(200 * scale),
                int(50 * scale), 1.0 * scale, max(1, int(scale)), _TEAL)

    # --- Secondary metrics (top-left, smaller) ---
    lines: list[tuple[str, tuple]] = [
        (f"Conf: {int(conf * 100)}%{'  (held)' if is_held else ''}", conf_color),
    ]

    kfl = f"{metrics.knee_flexion_L:.0f}deg" if metrics.knee_flexion_L else "---"
    kfr = f"{metrics.knee_flexion_R:.0f}deg" if metrics.knee_flexion_R else "---"
    lines.append((f"Knee  L:{kfl} R:{kfr}", _WHITE))

    if metrics.knee_flexion_diff is not None:
        lines.append((f"Sym: +/-{metrics.knee_flexion_diff:.0f}deg", _WHITE))

    if metrics.lean_angle_deg is not None and metrics.edge_angle_deg is not None:
        lines.append((f"Lean: {metrics.lean_angle_deg:.0f}deg", _WHITE))

    if turn_idx is not None and turn_side is not None:
        lines.append((f"Turn {turn_idx + 1}: {turn_side}", _TEAL))

    if metrics.overall_score is not None:
        sc_color = (0, 200, 80) if metrics.overall_score >= 65 else (
            _ORANGE_WARN if metrics.overall_score < 50 else _WHITE)
        lines.append((f"Score: {metrics.overall_score:.0f}/100", sc_color))

    # Draw background for HUD text block
    bg_w = int(220 * scale)
    bg_h = int(len(lines) * line_h + 10 * scale)
    overlay_bg = frame[max(0, y - int(20 * scale)):y + bg_h,
                       hud_x:hud_x + bg_w].copy()
    cv2.rectangle(frame, (hud_x, max(0, y - int(20 * scale))),
                  (hud_x + bg_w, y + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bg, 0.35,
                    frame[max(0, y - int(20 * scale)):y + bg_h,
                          hud_x:hud_x + bg_w],
                    0.65, 0,
                    frame[max(0, y - int(20 * scale)):y + bg_h,
                          hud_x:hud_x + bg_w])

    ts = 0.52 * scale
    tt = max(1, int(scale))
    for text, color in lines:
        _shadow(frame, text, hud_x + int(6 * scale), y, ts, tt, color)
        y += line_h

    # LOW CONFIDENCE banner — only when truly no skeleton (pose=None),
    # not when we have a gap-filled or low-conf partial pose
    if pose is None and conf < 0.2:
        text = "NO DETECTION"
        ts2 = 0.9 * scale
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts2, 2)
        _shadow(frame, text, w_frame // 2 - tw // 2, int(45 * scale),
                ts2, 2, _ORANGE_WARN)


def _find_current_turn(ts: float, turns: list[TurnSummary]):
    for t in turns:
        if t.start_s <= ts <= t.end_s:
            return t.turn_idx, t.side
    return None, None


def _resize_frame(frame: np.ndarray, max_dim: int | None) -> np.ndarray:
    if not max_dim:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    s = max_dim / max(h, w)
    return cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def _interpolate_pose(a: FramePose, b: FramePose, alpha: float) -> FramePose:
    alpha = max(0.0, min(1.0, alpha))
    ia = 1.0 - alpha
    n = min(len(a.landmarks), len(b.landmarks))
    lms = [
        PoseLandmark(
            x=ia * a.landmarks[i].x + alpha * b.landmarks[i].x,
            y=ia * a.landmarks[i].y + alpha * b.landmarks[i].y,
            z=ia * a.landmarks[i].z + alpha * b.landmarks[i].z,
            visibility=min(a.landmarks[i].visibility, b.landmarks[i].visibility),
        )
        for i in range(n)
    ]
    # Interpolate bbox
    bbox = None
    if a.detection_bbox and b.detection_bbox:
        ax1, ay1, ax2, ay2 = a.detection_bbox
        bx1, by1, bx2, by2 = b.detection_bbox
        bbox = (
            int(ia * ax1 + alpha * bx1), int(ia * ay1 + alpha * by1),
            int(ia * ax2 + alpha * bx2), int(ia * ay2 + alpha * by2),
        )
    elif a.detection_bbox:
        bbox = a.detection_bbox
    return FramePose(
        frame_idx=a.frame_idx,
        timestamp_s=ia * a.timestamp_s + alpha * b.timestamp_s,
        landmarks=lms,
        pose_confidence=ia * a.pose_confidence + alpha * b.pose_confidence,
        is_smoothed=True,
        world_landmarks=None,
        tracking_quality=min(a.tracking_quality, b.tracking_quality),
        detection_bbox=bbox,
    )


def render_overlay_video(
    video_path: Path,
    poses: list[FramePose | None],
    metrics_list: list[FrameMetrics],
    turns: list[TurnSummary],
    output_path: Path,
    max_dimension: int | None = None,
    show_bbox: bool = False,
) -> str:
    """Render annotated overlay video. Returns codec string used."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = native_w, native_h
    if max_dimension:
        s = min(1.0, max_dimension / max(native_w, native_h))
        out_w = max(1, int(native_w * s))
        out_h = max(1, int(native_h * s))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec_used = "avc1"
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"avc1"),
                             native_fps, (out_w, out_h))
    if not writer.isOpened():
        codec_used = "mp4v"
        writer = cv2.VideoWriter(str(output_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 native_fps, (out_w, out_h))

    poses_by_frame: dict[int, FramePose] = {
        p.frame_idx: p for p in poses if p is not None
    }
    metrics_by_frame: dict[int, FrameMetrics] = {
        m.frame_idx: m for m in metrics_list
    }
    sampled_pose_idxs = sorted(poses_by_frame.keys())
    sampled_metrics_idxs = sorted(metrics_by_frame.keys())

    stride = 1
    if len(sampled_pose_idxs) >= 2:
        strides = [sampled_pose_idxs[i + 1] - sampled_pose_idxs[i]
                   for i in range(len(sampled_pose_idxs) - 1)]
        stride = max(1, int(np.median(strides)))

    hud_s = _hud_scale(out_w, out_h)
    frame_idx = 0

    # Scale factor for the bbox (analysis vs native resolution)
    # analysis was at max_dimension, native might be larger
    bbox_scale_x = out_w / (max_dimension if max_dimension else native_w)
    bbox_scale_y = out_h / (max_dimension if max_dimension else native_h)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _resize_frame(frame, max_dimension)
            h, w = frame.shape[:2]

            # Interpolate between prev/next sampled poses
            pose: FramePose | None = None
            is_held = False

            if sampled_pose_idxs:
                pos = bisect.bisect_right(sampled_pose_idxs, frame_idx) - 1
                if pos >= 0 and sampled_pose_idxs[pos] == frame_idx:
                    pose = poses_by_frame[frame_idx]
                elif pos >= 0:
                    prev_idx = sampled_pose_idxs[pos]
                    next_pos = pos + 1
                    if next_pos < len(sampled_pose_idxs):
                        next_idx = sampled_pose_idxs[next_pos]
                        alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
                        pose = _interpolate_pose(
                            poses_by_frame[prev_idx],
                            poses_by_frame[next_idx],
                            alpha,
                        )
                    else:
                        stale = frame_idx - prev_idx
                        if stale <= stride:
                            pose = poses_by_frame[prev_idx]
                            is_held = True

            metrics: FrameMetrics | None = None
            if sampled_metrics_idxs:
                pos = bisect.bisect_right(sampled_metrics_idxs, frame_idx) - 1
                if pos >= 0:
                    last_idx = sampled_metrics_idxs[pos]
                    if frame_idx - last_idx <= stride:
                        metrics = metrics_by_frame[last_idx]

            # --- Draw ---
            if pose is not None:
                dim = _STALE_DIM if is_held else 1.0
                if show_bbox:
                    _draw_bbox(frame, pose, hud_s)
                _draw_skeleton(frame, pose, w, h, dim=dim)

            turn_idx, turn_side = None, None
            if metrics is not None:
                turn_idx, turn_side = _find_current_turn(
                    metrics.timestamp_s, turns
                )
            _draw_hud(frame, metrics, pose, turn_idx, turn_side,
                      scale=hud_s, is_held=is_held)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return codec_used
