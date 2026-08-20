"""Score-reliability and quality-warning helpers (Improvements V2, Phase 2).

The orchestrator builds a :class:`QualityReport` at the end of every run. This
module owns the *content* of the new public reliability contract: which
warning codes fire, which `score_reliability` tier is reported, whether the
score counts for progress trends, and whether stance was measurable.

Signals consumed are deliberately limited to what the current pipeline can
already produce — pose confidence, ankle-visibility (used by `viewpoint.py`),
turn count, tracking-segment count, and the existing per-frame metrics
(`hip_tilt`, `com_shift_x`). When a signal isn't available, the relevant
warning code is *not* emitted; this module never invents data.

See:
- `communication/README.md` for the shared cross-agent contract.
- `improvement_proposal/metrics_and_coaching_plan.md` (Phases 2 & 5) for the
  pre-harness starting thresholds calibrated here.
"""

from __future__ import annotations

import numpy as np

from technique_analysis.common.contracts.models import (
    FrameMetrics,
    FramePose,
    TrackingSegment,
    TurnSummary,
)

# MediaPipe landmark indices for ankles. Matches viewpoint.detect_viewpoint.
_L_ANKLE = 27
_R_ANKLE = 28

# Per-landmark visibility floor that counts as "visible enough to measure stance".
_STANCE_VIS_THRESHOLD = 0.5

# Fraction-of-frames threshold below which stance is considered not measurable.
# Matches the existing viewpoint heuristic's `_BOTH_VISIBLE_FRACTION` so the
# new structured field tracks the long-standing warning string behavior.
_STANCE_MEASURABLE_FRACTION = 0.30

# Pose-confidence thresholds for the public warnings.
_POSE_CONF_LOW_MEAN = 0.50
_POSE_CONF_INSUFFICIENT_MEAN = 0.35
_LOW_CONF_FRACTION_WARN = 0.30
_LOW_CONF_FRACTION_INSUFFICIENT = 0.60

# Turn-level thresholds.
_SHORT_CLIP_S = 4.0
_LOW_TURN_BOUNDARY_CONF = 0.50

# Wedge / snowplow heuristic — see metrics_and_coaching_plan.md Phase 5.
# Starting thresholds (calibration rows 17, 18). The plan documents these as
# "starting values" pinned by Phase 0 harness; we use them conservatively here.
_WEDGE_T_HIP_MIN_DEG = 0.5       # = 0.5 × segmenter _MIN_AMPLITUDE_HIP (1.0 deg)
_WEDGE_T_COM_MAX_METRES = 0.005  # = 0.5 × segmenter _MIN_AMPLITUDE_COM (0.01 m)

# Allowed values per communication/README.md "Shared Contract".
RELIABILITY_RELIABLE = "reliable"
RELIABILITY_LIMITED = "limited"
RELIABILITY_INSUFFICIENT = "insufficient"

# Warning-code constants. Identical strings appear in communication/README.md
# and the Codex TS-mirror in MVP/web/lib/analysis-summary.ts (added by Codex).
WARN_LOW_POSE_CONFIDENCE = "low_pose_confidence"
WARN_INSUFFICIENT_SKELETON = "insufficient_skeleton_detection"
WARN_STANCE_NOT_MEASURABLE = "stance_not_measurable"
WARN_WEDGE_LIKELY = "wedge_likely"
WARN_SHORT_CLIP = "short_clip"
WARN_LOW_BOUNDARY = "low_boundary_reliability"
WARN_TRACKING_LOSS = "tracking_loss"
# WARN_FOLLOW_CAM_DEGRADED is intentionally *not* emitted in this pass; the
# camera-motion signal it would need isn't available yet. Documented in
# communication/claude_status.md and communication/backend_contract.md.


def compute_stance_visibility_fraction(poses: list[FramePose | None]) -> float:
    """Return the fraction of analysis frames where both ankles are visible.

    "Visible" matches viewpoint.detect_viewpoint: both ankle landmarks present
    with visibility >= 0.5. Returns 1.0 when there are no usable poses (no
    evidence of degradation; the broader pipeline will already have set
    stronger warnings such as `insufficient_skeleton_detection`).
    """
    valid = [p for p in poses if p is not None]
    if not valid:
        return 1.0
    both_visible = 0
    for p in valid:
        lms = p.landmarks
        if len(lms) > max(_L_ANKLE, _R_ANKLE):
            if (
                lms[_L_ANKLE].visibility >= _STANCE_VIS_THRESHOLD
                and lms[_R_ANKLE].visibility >= _STANCE_VIS_THRESHOLD
            ):
                both_visible += 1
    return both_visible / len(valid)


def compute_wedge_likely(
    metrics_list: list[FrameMetrics],
    stance_measurable: bool,
) -> bool:
    """Heuristic flag: hip rotation visible while CoM commitment is absent.

    A carved turn produces lateral CoM excursion *and* hip rotation. A wedge /
    snowplow rotates the hips over wedged skis without committing weight
    laterally. Detect that pattern from the existing per-frame signals.

    Conservative implementation:
    - Only fires when stance is measurable (need ankles to call this).
    - Only fires when at least ~10 high-confidence frames carry both signals
      (avoid noisy short-clip false positives).
    - Uses `std` as the amplitude proxy on confident frames. Peak-to-peak is
      tempting but it's dominated by outliers on noisy monocular signals.

    The thresholds are the plan's starting values (rows 17, 18); they are
    `harness-pin candidates`, not validated values.
    """
    if not stance_measurable:
        return False

    confident = [m for m in metrics_list if m.pose_confidence >= 0.4]
    hip_vals = [m.hip_tilt for m in confident if m.hip_tilt is not None]
    com_vals = [m.com_shift_x for m in confident if m.com_shift_x is not None]
    if len(hip_vals) < 10 or len(com_vals) < 10:
        return False

    hip_amp = float(np.std(hip_vals))
    com_amp = float(np.std(com_vals))
    return hip_amp > _WEDGE_T_HIP_MIN_DEG and com_amp < _WEDGE_T_COM_MAX_METRES


def _avg_turn_pose_confidence(turns: list[TurnSummary]) -> float | None:
    if not turns:
        return None
    confs = [t.avg_pose_confidence for t in turns if t.avg_pose_confidence is not None]
    return float(np.mean(confs)) if confs else None


def compute_reliability(
    *,
    low_confidence_fraction: float,
    overall_pose_confidence_mean: float,
    stance_visibility_fraction: float,
    stance_measurable: bool,
    wedge_likely: bool,
    turns: list[TurnSummary],
    segments: list[TrackingSegment],
    video_duration_s: float,
) -> tuple[str, bool, list[str]]:
    """Return ``(score_reliability, score_counts_for_progress, warning_codes)``.

    The contract is:

    - ``score_reliability``:
        * ``"insufficient"`` when skeleton detection itself is too weak to
          trust, *or* the segmenter couldn't find any turns on a long-enough
          clip that also has low pose confidence.
        * ``"limited"`` when any softer warning code fires (low pose, stance
          unmeasurable, wedge, short clip, tracking loss, low boundary
          reliability).
        * ``"reliable"`` otherwise.
    - ``score_counts_for_progress`` is ``False`` only when reliability is
      ``"insufficient"``. This matches the README rule and the Codex assumption.
    - ``warning_codes`` is the union of fired codes from the documented set.
    """
    warnings: list[str] = []

    # --- Skeleton / pose confidence ----------------------------------------
    insufficient_skeleton = (
        overall_pose_confidence_mean < _POSE_CONF_INSUFFICIENT_MEAN
        or low_confidence_fraction > _LOW_CONF_FRACTION_INSUFFICIENT
    )
    if insufficient_skeleton:
        warnings.append(WARN_INSUFFICIENT_SKELETON)
    elif (
        overall_pose_confidence_mean < _POSE_CONF_LOW_MEAN
        or low_confidence_fraction > _LOW_CONF_FRACTION_WARN
    ):
        warnings.append(WARN_LOW_POSE_CONFIDENCE)

    # --- Stance ------------------------------------------------------------
    if not stance_measurable:
        warnings.append(WARN_STANCE_NOT_MEASURABLE)

    # --- Wedge -------------------------------------------------------------
    if wedge_likely:
        warnings.append(WARN_WEDGE_LIKELY)

    # --- Clip length ------------------------------------------------------
    if video_duration_s < _SHORT_CLIP_S:
        warnings.append(WARN_SHORT_CLIP)

    # --- Boundary reliability ---------------------------------------------
    # Two failure modes:
    #   (a) zero turns despite a long-enough clip and usable pose confidence,
    #   (b) turns detected but their per-turn pose confidence is low.
    avg_turn_conf = _avg_turn_pose_confidence(turns)
    if (
        not turns
        and video_duration_s >= _SHORT_CLIP_S
        and not insufficient_skeleton
    ):
        warnings.append(WARN_LOW_BOUNDARY)
    elif (
        avg_turn_conf is not None
        and avg_turn_conf < _LOW_TURN_BOUNDARY_CONF
    ):
        warnings.append(WARN_LOW_BOUNDARY)

    # --- Tracking loss / multi-segment ------------------------------------
    if len(segments) > 1:
        warnings.append(WARN_TRACKING_LOSS)

    # --- Tier classification ----------------------------------------------
    if WARN_INSUFFICIENT_SKELETON in warnings:
        reliability = RELIABILITY_INSUFFICIENT
    elif (
        not turns
        and video_duration_s >= _SHORT_CLIP_S
        and WARN_LOW_POSE_CONFIDENCE in warnings
    ):
        # No turns + low confidence on a long-enough clip ⇒ scoring evidence
        # is too thin to count for progress, even if no single signal hit the
        # insufficient threshold on its own.
        reliability = RELIABILITY_INSUFFICIENT
    elif warnings:
        reliability = RELIABILITY_LIMITED
    else:
        reliability = RELIABILITY_RELIABLE

    counts_for_progress = reliability != RELIABILITY_INSUFFICIENT
    return reliability, counts_for_progress, warnings
