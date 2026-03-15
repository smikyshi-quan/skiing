"""EMA landmark smoother and jitter scorer."""

from __future__ import annotations

import numpy as np

from technique_analysis.common.contracts.models import FramePose, PoseLandmark


class LandmarkSmoother:
    """Per-landmark EMA smoother over (x, y, z) coordinates.

    Smooths both 2D normalised landmarks and 3D world landmarks when present.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self._alpha = alpha
        self._state: list[tuple[float, float, float]] | None = None
        self._world_state: list[tuple[float, float, float]] | None = None

    def smooth(self, pose: FramePose) -> FramePose:
        """Return a new FramePose with EMA-smoothed landmark positions."""
        a = self._alpha

        # --- Smooth 2D landmarks ---
        if self._state is None:
            self._state = [(lm.x, lm.y, lm.z) for lm in pose.landmarks]
            smoothed_landmarks = pose.landmarks
        else:
            new_state: list[tuple[float, float, float]] = []
            smoothed_landmarks = []
            for i, lm in enumerate(pose.landmarks):
                if i < len(self._state):
                    px, py, pz = self._state[i]
                    sx = a * lm.x + (1 - a) * px
                    sy = a * lm.y + (1 - a) * py
                    sz = a * lm.z + (1 - a) * pz
                else:
                    sx, sy, sz = lm.x, lm.y, lm.z
                new_state.append((sx, sy, sz))
                smoothed_landmarks.append(
                    PoseLandmark(x=sx, y=sy, z=sz, visibility=lm.visibility)
                )
            self._state = new_state

        # --- Smooth 3D world landmarks (if present) ---
        smoothed_world: list[PoseLandmark] | None = None
        if pose.world_landmarks is not None:
            if self._world_state is None:
                self._world_state = [(lm.x, lm.y, lm.z) for lm in pose.world_landmarks]
                smoothed_world = pose.world_landmarks
            else:
                new_world_state: list[tuple[float, float, float]] = []
                smoothed_world = []
                for i, lm in enumerate(pose.world_landmarks):
                    if i < len(self._world_state):
                        px, py, pz = self._world_state[i]
                        sx = a * lm.x + (1 - a) * px
                        sy = a * lm.y + (1 - a) * py
                        sz = a * lm.z + (1 - a) * pz
                    else:
                        sx, sy, sz = lm.x, lm.y, lm.z
                    new_world_state.append((sx, sy, sz))
                    smoothed_world.append(
                        PoseLandmark(x=sx, y=sy, z=sz, visibility=lm.visibility)
                    )
                self._world_state = new_world_state

        return FramePose(
            frame_idx=pose.frame_idx,
            timestamp_s=pose.timestamp_s,
            landmarks=smoothed_landmarks,
            pose_confidence=pose.pose_confidence,
            is_smoothed=True,
            world_landmarks=smoothed_world,
            tracking_quality=pose.tracking_quality,
            detection_bbox=pose.detection_bbox,
        )


def compute_jitter_score(poses: list[FramePose]) -> float:
    """Mean frame-to-frame landmark velocity, normalized by torso size."""
    if len(poses) < 2:
        return 0.0

    valid_poses = [p for p in poses if p is not None and len(p.landmarks) >= 29]
    if len(valid_poses) < 2:
        return 0.0

    # Estimate torso size from shoulder-hip distance (landmark 11/12 to 23/24)
    torso_sizes = []
    for p in valid_poses:
        lms = p.landmarks
        if len(lms) > 24:
            mid_shoulder_y = (lms[11].y + lms[12].y) / 2
            mid_hip_y = (lms[23].y + lms[24].y) / 2
            torso_sizes.append(abs(mid_hip_y - mid_shoulder_y))
    torso_size = float(np.median(torso_sizes)) if torso_sizes else 0.1
    if torso_size < 1e-6:
        torso_size = 0.1

    velocities = []
    for i in range(1, len(valid_poses)):
        p0 = valid_poses[i - 1]
        p1 = valid_poses[i]
        n = min(len(p0.landmarks), len(p1.landmarks))
        for j in range(n):
            dx = p1.landmarks[j].x - p0.landmarks[j].x
            dy = p1.landmarks[j].y - p0.landmarks[j].y
            velocities.append(np.sqrt(dx * dx + dy * dy))

    return float(np.mean(velocities)) / torso_size if velocities else 0.0
