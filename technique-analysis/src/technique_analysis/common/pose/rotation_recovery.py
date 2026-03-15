"""Rotation-augmented keypoint recovery for low-confidence frames.

Based on Zwölfer et al. 2023 (Nature Scientific Reports):
  "A graph-based approach can improve keypoint detection of complex poses:
   a proof-of-concept on injury occurrences in alpine ski racing"

When MediaPipe confidence falls below CONF_TRIGGER (e.g. during pole-plant
moments or out-of-balance poses), this module rotates the crop at incremental
angles, re-runs inference on each rotation, rotates landmarks back to the
original plane, then selects the best-confidence candidate per joint.

This is a per-frame simplified version of the paper's full temporal Dijkstra
optimizer.  The temporal version optimises across all frames simultaneously
(offline batch), which is the medium-term upgrade.  The per-frame version
already captures the primary benefit: escaping the degenerate orientation that
causes MediaPipe to drop joints (e.g. a pole held across the body at 45°).

Integration:
    Called from PoseExtractor.extract() after the primary MediaPipe inference
    when pose_confidence < CONF_TRIGGER.  Operates on the person crop in
    crop-normalised coordinates, before _transform_landmarks converts to
    full-frame coords.
"""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from technique_analysis.common.contracts.models import PoseLandmark

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trigger recovery when overall pose confidence is below this.
CONF_TRIGGER: float = 0.55

# Rotation step in degrees.  36 steps @ 10° cover all orientations (paper default).
# Set to 30 for a faster 12-step pass; set to 10 for full paper fidelity.
STEP_DEG: int = 10

# Only replace a joint's landmark when the rotated candidate improves
# visibility by at least this amount over the original.
VISIBILITY_GAIN_MIN: float = 0.10

# Joints to attempt recovery on (indices into MediaPipe 33-point skeleton).
# Focus on the lower body and elbows — the joints most affected by poles and
# equipment occlusion in skiing.
_RECOVERY_JOINTS: frozenset[int] = frozenset([
    13, 14,   # elbows (arm-pole confusion)
    15, 16,   # wrists
    25, 26,   # knees  ← primary target (pole-plant occlusion)
    27, 28,   # ankles
    29, 30,   # heels
    31, 32,   # foot indices
])


# ---------------------------------------------------------------------------
# Image rotation helpers
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate *img* clockwise by *angle_deg* around its centre."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _rotate_landmark_back(
    x_norm: float, y_norm: float,
    angle_deg: float,
) -> tuple[float, float]:
    """Un-rotate a normalised (x, y) from the rotated frame back to original.

    Normalised coords have centre at (0.5, 0.5).  We rotate around that centre
    by the *negative* of *angle_deg* (inverse of the image rotation).
    """
    cx, cy = 0.5, 0.5
    px, py = x_norm - cx, y_norm - cy
    rad = math.radians(-angle_deg)          # inverse rotation
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx = cos_a * px - sin_a * py + cx
    ry = sin_a * px + cos_a * py + cy
    return rx, ry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recover_landmarks(
    crop_bgr: np.ndarray,
    run_mediapipe_fn: Callable[[np.ndarray], list[PoseLandmark] | None],
    original_landmarks: list[PoseLandmark],
) -> list[PoseLandmark]:
    """Return an improved landmark list by trying multiple crop rotations.

    For each joint in *_RECOVERY_JOINTS* whose visibility is below the
    improvement threshold, we search across all rotation candidates and
    replace it with the highest-visibility option found — provided the gain
    exceeds *VISIBILITY_GAIN_MIN*.

    All other joints (head, torso anchors) are left unchanged to keep the
    overall body position stable.

    Args:
        crop_bgr:          Person crop in BGR (crop-normalised coords).
        run_mediapipe_fn:  Callable that runs MediaPipe on a BGR crop and
                           returns a list of PoseLandmark in crop-normalised
                           coords, or None on failure.
        original_landmarks: Landmarks from the primary (0°) inference.

    Returns:
        A new landmark list of the same length as *original_landmarks*, with
        low-confidence recovery joints replaced where a better rotation was found.
    """
    n = len(original_landmarks)
    # best_candidates[i] = (visibility, x, y, z) for joint i
    best: list[tuple[float, float, float, float]] = [
        (lm.visibility, lm.x, lm.y, lm.z)
        for lm in original_landmarks
    ]

    rotations = range(STEP_DEG, 360, STEP_DEG)  # skip 0° — already have it
    for angle in rotations:
        rotated = _rotate_image(crop_bgr, angle)
        lms = run_mediapipe_fn(rotated)
        if lms is None or len(lms) < n:
            continue

        for i in lms if False else range(min(n, len(lms))):
            if i not in _RECOVERY_JOINTS:
                continue
            lm = lms[i]
            if lm.visibility <= best[i][0] + VISIBILITY_GAIN_MIN:
                continue  # not meaningfully better
            # Rotate landmark back to original orientation
            rx, ry = _rotate_landmark_back(lm.x, lm.y, angle)
            # Clip to [0, 1] — rotation can push edge landmarks slightly outside
            rx = max(0.0, min(1.0, rx))
            ry = max(0.0, min(1.0, ry))
            best[i] = (lm.visibility, rx, ry, lm.z)

    return [
        PoseLandmark(
            x=best[i][1],
            y=best[i][2],
            z=best[i][3],
            visibility=best[i][0],
        )
        for i in range(n)
    ]
