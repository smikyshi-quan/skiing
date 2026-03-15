"""Kalman-based primary person tracker for robust skier identification."""

from __future__ import annotations

import numpy as np

from technique_analysis.common.contracts.models import PoseLandmark

_L_HIP = 23
_R_HIP = 24
_HIP_VIS_MIN = 0.25


def _hip_midpoint(landmarks: list[PoseLandmark]) -> tuple[float, float] | None:
    if len(landmarks) <= _R_HIP:
        return None
    lh = landmarks[_L_HIP]
    rh = landmarks[_R_HIP]
    if lh.visibility < _HIP_VIS_MIN or rh.visibility < _HIP_VIS_MIN:
        return None
    return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)


def _bbox_area(landmarks: list[PoseLandmark]) -> float:
    if not landmarks:
        return 0.0
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


class PersonTracker:
    """Single-person Kalman tracker on hip midpoint position.

    State vector: [hip_x, hip_y, vx, vy] in normalized image coordinates.
    Uses a constant-velocity motion model.

    When multiple detections are present, selects the one closest to the
    predicted position (falling back to largest bounding box when no hip
    match is reliable). This prevents the pipeline from accidentally
    switching to a bystander when the primary skier is briefly occluded.
    """

    def __init__(self, match_threshold: float = 0.30) -> None:
        """
        Args:
            match_threshold: Maximum distance (normalized coords) to accept
                a hip measurement as belonging to the tracked person.
                If the closest detection is farther than this, fall back to
                largest-bbox selection (but still update Kalman with it).
        """
        self._match_threshold = match_threshold
        self._initialized = False

        # State: [hip_x, hip_y, vx, vy]
        self._x = np.zeros(4)
        # Covariance — start with moderate uncertainty
        self._P = np.eye(4) * 0.05

        # Process noise Q: positional uncertainty grows slowly, velocity faster
        self._Q = np.diag([2e-4, 2e-4, 5e-3, 5e-3])
        # Measurement noise R: MediaPipe hip position is fairly stable
        self._R = np.diag([3e-4, 3e-4])
        # Observation matrix: we observe x and y only
        self._H = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]])

    # ------------------------------------------------------------------
    # Private Kalman steps
    # ------------------------------------------------------------------

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _predict(self, dt: float) -> np.ndarray:
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q
        return self._x[:2]

    def _update(self, z: tuple[float, float]) -> None:
        z_vec = np.array(z)
        innov = z_vec - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        self._P = (np.eye(4) - K @ self._H) @ self._P

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_best_bbox(
        self,
        bboxes: list[tuple[int, int, int, int, float]],
        frame_w: int,
        frame_h: int,
        dt: float = 0.05,
    ) -> int:
        """Select best bounding box from YOLO detections using Kalman tracking.

        Args:
            bboxes: Non-empty list of (x1, y1, x2, y2, conf) detections.
            frame_w/h: Frame dimensions for normalising coordinates.
            dt: Elapsed time since last call in seconds.

        Returns:
            Index into *bboxes* for the selected person.
        """
        if not bboxes:
            raise ValueError("bboxes must be non-empty")

        # Normalise bbox centers for tracking
        centers = [
            ((x1 + x2) / 2.0 / frame_w, (y1 + y2) / 2.0 / frame_h)
            for x1, y1, x2, y2, _ in bboxes
        ]
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _ in bboxes]

        if not self._initialized:
            best = max(range(len(bboxes)), key=lambda i: areas[i])
            cx, cy = centers[best]
            self._x = np.array([cx, cy, 0.0, 0.0])
            self._P = np.diag([1e-3, 1e-3, 1e-2, 1e-2])
            self._initialized = True
            return best

        predicted = self._predict(dt)

        best_i = min(
            range(len(centers)),
            key=lambda i: float(np.linalg.norm(np.array(centers[i]) - predicted)),
        )
        best_dist = float(np.linalg.norm(np.array(centers[best_i]) - predicted))

        if best_dist > self._match_threshold:
            best_i = max(range(len(bboxes)), key=lambda i: areas[i])

        self._update(centers[best_i])
        return best_i

    def select_best(
        self,
        detections: list[list[PoseLandmark]],
        dt: float = 0.05,
    ) -> int:
        """Return index of the best detection from *detections*.

        On the first call, bootstraps using the largest bounding box.
        Subsequent calls use Kalman prediction to pick the closest match.

        Args:
            detections: Non-empty list of landmark lists (one per detected person).
            dt: Elapsed time since last call in seconds (used for prediction).

        Returns:
            Index into *detections* for the selected person.
        """
        if not detections:
            raise ValueError("detections must be non-empty")

        if not self._initialized:
            # Bootstrap: pick largest bbox
            best = max(range(len(detections)), key=lambda i: _bbox_area(detections[i]))
            hip = _hip_midpoint(detections[best])
            if hip is not None:
                self._x = np.array([hip[0], hip[1], 0.0, 0.0])
                self._P = np.diag([1e-3, 1e-3, 1e-2, 1e-2])
                self._initialized = True
            return best

        predicted = self._predict(dt)

        # Associate: find detection with hip closest to predicted hip
        best_i = -1
        best_dist = float("inf")
        for i, lms in enumerate(detections):
            hip = _hip_midpoint(lms)
            if hip is None:
                continue
            dist = float(np.linalg.norm(np.array(hip) - predicted))
            if dist < best_dist:
                best_dist = dist
                best_i = i

        if best_i < 0 or best_dist > self._match_threshold:
            # No reliable hip match → fall back to largest bbox
            best_i = max(range(len(detections)), key=lambda i: _bbox_area(detections[i]))

        # Update Kalman with selected detection's hip
        hip = _hip_midpoint(detections[best_i])
        if hip is not None:
            self._update(hip)

        return best_i
