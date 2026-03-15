"""Skeleton-topology joint refiner for occluded / low-confidence joints.

Motivation
----------
After MediaPipe inference (and optional rotation recovery), certain joints
may still have low visibility — the knees during pole-plant being the primary
skiing example.  A full Graph Convolutional Network (GCN) would learn to infer
these positions from the rest of the skeleton; this module provides an
anatomically-grounded approximation without requiring a pre-trained model.

For each joint below *VIS_THRESHOLD*, we check whether both its kinematic
parent and its distal child in the skeleton chain are visible.  If they are,
the joint is estimated by linear interpolation at the known bone-length ratio
for that segment.  The estimated position is blended with the original
(low-confidence) position using a blend weight that scales with the confidence
gap, so high-confidence joints are never moved.

Bone-length ratios
------------------
Based on Winter's anthropometric tables (2009) and MediaPipe 3D benchmarks:
  - Knee sits 52% of the way from hip to ankle along the leg chain.
  - Elbow sits 50% of the way from shoulder to wrist.

These ratios apply to both 2D (image-normalised) and 3D world landmarks.

MediaPipe 33-point joint indices used here
------------------------------------------
  11 L shoulder   12 R shoulder
  13 L elbow      14 R elbow
  15 L wrist      16 R wrist
  23 L hip        24 R hip
  25 L knee       26 R knee
  27 L ankle      28 R ankle
  29 L heel       30 R heel
  31 L foot_idx   32 R foot_idx
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np

from technique_analysis.common.contracts.models import FramePose, PoseLandmark

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Joints below this visibility are candidates for refinement.
VIS_THRESHOLD: float = 0.40

# When blending estimated vs. original position, this is the maximum weight
# given to the estimate.  1.0 = fully replace; 0.8 = leave 20% of original.
MAX_BLEND: float = 0.85

# ---------------------------------------------------------------------------
# Kinematic chains: (parent_idx, target_idx, child_idx, t_ratio)
#
# t_ratio = where target sits along the parent→child_end line.
# Estimated position = parent + t * (child_end - parent)
# ---------------------------------------------------------------------------

_CHAINS: tuple[tuple[int, int, int, float], ...] = (
    # Legs (most important for skiing metrics)
    (23, 25, 27, 0.52),   # L hip  → L knee  → L ankle,  knee at 52%
    (24, 26, 28, 0.52),   # R hip  → R knee  → R ankle,  knee at 52%
    (25, 27, 29, 0.55),   # L knee → L ankle → L heel,   ankle at 55%
    (26, 28, 30, 0.55),   # R knee → R ankle → R heel,   ankle at 55%
    # Arms (secondary — elbow occlusion by poles)
    (11, 13, 15, 0.50),   # L shoulder → L elbow → L wrist, elbow at 50%
    (12, 14, 16, 0.50),   # R shoulder → R elbow → R wrist, elbow at 50%
)


# ---------------------------------------------------------------------------
# Per-landmark interpolation helpers
# ---------------------------------------------------------------------------

def _interpolate(
    parent: PoseLandmark,
    child_end: PoseLandmark,
    t: float,
) -> tuple[float, float, float]:
    """Linear interpolation between parent and child_end at ratio t."""
    x = parent.x + t * (child_end.x - parent.x)
    y = parent.y + t * (child_end.y - parent.y)
    z = parent.z + t * (child_end.z - parent.z)
    return x, y, z


def _refine_list(
    landmarks: list[PoseLandmark],
) -> list[PoseLandmark]:
    """Apply chain-based refinement to a landmark list (2D or 3D coords)."""
    if len(landmarks) < 33:
        return landmarks

    result = list(landmarks)
    for parent_i, target_i, child_i, t in _CHAINS:
        target = result[target_i]
        if target.visibility >= VIS_THRESHOLD:
            continue  # already confident — don't touch it

        parent_lm = result[parent_i]
        child_lm  = result[child_i]
        if parent_lm.visibility < VIS_THRESHOLD or child_lm.visibility < VIS_THRESHOLD:
            continue  # can't infer without both anchors being reliable

        ex, ey, ez = _interpolate(parent_lm, child_lm, t)

        # Blend: the lower the original visibility, the more we trust the estimate.
        # confidence_gap in [0, VIS_THRESHOLD] → blend in [0, MAX_BLEND]
        gap = VIS_THRESHOLD - target.visibility
        blend = min(MAX_BLEND, gap / VIS_THRESHOLD * MAX_BLEND)

        nx = blend * ex + (1 - blend) * target.x
        ny = blend * ey + (1 - blend) * target.y
        nz = blend * ez + (1 - blend) * target.z

        # Inferred visibility: geometric mean of anchors, reduced by blend factor
        inferred_vis = blend * float(np.sqrt(parent_lm.visibility * child_lm.visibility))
        new_vis = max(target.visibility, inferred_vis)

        result[target_i] = PoseLandmark(x=nx, y=ny, z=nz, visibility=new_vis)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SkeletalRefiner:
    """Stateless post-processor that infers occluded joint positions.

    Usage in the pipeline (orchestrator.py)::

        refiner = SkeletalRefiner()
        ...
        pose = smoother.smooth(pose)
        pose = refiner.refine(pose)   # ← add after smoother
    """

    def refine(self, pose: FramePose) -> FramePose:
        """Return a new FramePose with low-visibility joints inferred where possible.

        Both 2D (image-normalised) and 3D (world) landmarks are refined
        independently using the same chain topology.
        """
        refined_2d = _refine_list(pose.landmarks)

        refined_world: list[PoseLandmark] | None = None
        if pose.world_landmarks is not None:
            refined_world = _refine_list(pose.world_landmarks)

        if refined_2d is pose.landmarks and refined_world is pose.world_landmarks:
            return pose  # nothing changed

        return dc_replace(
            pose,
            landmarks=refined_2d,
            world_landmarks=refined_world,
        )
