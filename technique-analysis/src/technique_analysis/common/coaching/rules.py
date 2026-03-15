"""Rule-based, deterministic coaching tip generator."""

from __future__ import annotations

import numpy as np

from technique_analysis.common.contracts.models import (
    CoachingTip,
    FrameMetrics,
    QualityReport,
    TurnSummary,
)


def _non_none(values: list) -> list:
    return [v for v in values if v is not None]


def generate_coaching_tips(
    metrics_list: list[FrameMetrics],
    turns: list[TurnSummary],
    quality: QualityReport,
) -> list[CoachingTip]:
    """Return 0–7 coaching tips sorted by severity (action > warn > info)."""
    tips: list[CoachingTip] = []

    if metrics_list:
        conf_mask = [m for m in metrics_list if m.pose_confidence >= 0.4]
    else:
        conf_mask = []

    # Rule 1: symmetric knee flexion (action if > 15° average diff)
    diffs = _non_none([m.knee_flexion_diff for m in conf_mask])
    if diffs:
        mean_diff = float(np.mean(diffs))
        if mean_diff > 15.0:
            tips.append(CoachingTip(
                title="Work on symmetric knee flexion",
                explanation=(
                    f"Average left-right knee flexion asymmetry is {mean_diff:.1f}°. "
                    "Aim for <10° symmetry to distribute load evenly and improve edge control."
                ),
                evidence=f"knee_flexion_diff mean={mean_diff:.1f}°",
                severity="action",
            ))

    # Rule 2: upper body rotation (warn if shoulder_tilt std > 8°)
    shoulder_tilts = _non_none([m.shoulder_tilt for m in conf_mask])
    if shoulder_tilts:
        std_tilt = float(np.std(shoulder_tilts))
        if std_tilt > 8.0:
            tips.append(CoachingTip(
                title="Quiet your upper body rotation",
                explanation=(
                    f"Shoulder tilt standard deviation is {std_tilt:.1f}°. "
                    "Excessive upper body rotation wastes energy; focus on separating "
                    "hip and shoulder movement."
                ),
                evidence=f"shoulder_tilt std={std_tilt:.1f}°",
                severity="warn",
            ))

    # Rule 3: stance width
    widths = _non_none([m.stance_width_ratio for m in conf_mask])
    if widths:
        mean_width = float(np.mean(widths))
        if mean_width < 0.8:
            tips.append(CoachingTip(
                title="Widen your stance",
                explanation=(
                    f"Average stance width ratio is {mean_width:.2f} (ankles relative to hips). "
                    "A narrower stance reduces lateral stability. Try shoulder-width spacing."
                ),
                evidence=f"stance_width_ratio mean={mean_width:.2f}",
                severity="warn",
            ))
        elif mean_width > 2.0:
            tips.append(CoachingTip(
                title="Narrow your stance slightly",
                explanation=(
                    f"Average stance width ratio is {mean_width:.2f}. "
                    "An overly wide stance can hinder clean edge-to-edge transitions."
                ),
                evidence=f"stance_width_ratio mean={mean_width:.2f}",
                severity="warn",
            ))

    # Rule 4: upper body quietness
    quietness_vals = _non_none([m.upper_body_quietness for m in conf_mask])
    if quietness_vals:
        mean_quietness = float(np.mean(quietness_vals))
        threshold = 1e-4
        if mean_quietness > threshold:
            tips.append(CoachingTip(
                title="Stabilise your upper body",
                explanation=(
                    f"Upper-body quietness score {mean_quietness:.2e} indicates notable "
                    "head/torso movement. Focus on keeping your core still while hips drive turns."
                ),
                evidence=f"upper_body_quietness mean={mean_quietness:.2e}",
                severity="info",
            ))

    # Rule 5: low confidence → camera framing
    if quality.low_confidence_fraction > 0.30:
        tips.append(CoachingTip(
            title="Improve camera angle for better analysis",
            explanation=(
                f"Pose was low-confidence in {quality.low_confidence_fraction:.0%} of frames. "
                "Position the camera to capture the full body from waist to feet."
            ),
            evidence=f"low_confidence_fraction={quality.low_confidence_fraction:.2f}",
            severity="info",
        ))

    # Rule 6: hip-knee-ankle alignment
    align_vals_L = _non_none([m.hip_knee_ankle_alignment_L for m in conf_mask])
    align_vals_R = _non_none([m.hip_knee_ankle_alignment_R for m in conf_mask])
    align_vals = align_vals_L + align_vals_R
    if align_vals:
        mean_align = float(np.mean(align_vals))
        if mean_align > 0.3:
            tips.append(CoachingTip(
                title="Drive your knees forward over your toes",
                explanation=(
                    f"Hip-knee-ankle alignment score is {mean_align:.2f} (0=ideal stack). "
                    "Pressing knees forward improves edge engagement and fore-aft balance."
                ),
                evidence=f"hip_knee_ankle_alignment mean={mean_align:.2f}",
                severity="warn",
            ))

    # Rule 7: overall score — low average score (action if <50, warn if <65)
    scored = _non_none([m.overall_score for m in conf_mask])
    if scored:
        mean_score = float(np.mean(scored))
        if mean_score < 50.0:
            tips.append(CoachingTip(
                title="Significant technique improvements needed",
                explanation=(
                    f"Average technique score is {mean_score:.0f}/100. "
                    "Multiple mechanics need attention — focus on knee flexion, "
                    "balance, and body alignment before adding speed."
                ),
                evidence=f"overall_score mean={mean_score:.1f}",
                severity="action",
            ))
        elif mean_score < 65.0:
            tips.append(CoachingTip(
                title="Technique has room to grow",
                explanation=(
                    f"Average technique score is {mean_score:.0f}/100. "
                    "Work on the specific metrics below to reach the 'Good' range."
                ),
                evidence=f"overall_score mean={mean_score:.1f}",
                severity="warn",
            ))

    # Rule 8 (was 7): edge angle (3D) — low edge angle indicates passive carving
    edge_vals = _non_none([m.edge_angle_deg for m in conf_mask])
    if edge_vals:
        mean_edge = float(np.mean(edge_vals))
        if mean_edge < 10.0:
            tips.append(CoachingTip(
                title="Increase your edge angle",
                explanation=(
                    f"Average lower-leg inclination is {mean_edge:.1f}° from vertical. "
                    "Greater edge angle (15–35°) generates more carving force and speed "
                    "through turns. Focus on driving the knees laterally into the hill."
                ),
                evidence=f"edge_angle_deg mean={mean_edge:.1f}°",
                severity="warn",
            ))

    # Rule 9 (was 8): lean angle (3D) — excessive lean indicates over-rotation or falling
    lean_vals = _non_none([m.lean_angle_deg for m in conf_mask])
    if lean_vals:
        mean_lean = float(np.mean(lean_vals))
        if mean_lean > 25.0:
            tips.append(CoachingTip(
                title="Reduce excessive body lean",
                explanation=(
                    f"Average full-body lean is {mean_lean:.1f}° from vertical. "
                    "Leaning beyond 20–25° risks losing balance. Lead with your hips "
                    "and keep your upper body more upright."
                ),
                evidence=f"lean_angle_deg mean={mean_lean:.1f}°",
                severity="warn",
            ))

    # Sort by severity
    severity_order = {"action": 0, "warn": 1, "info": 2}
    tips.sort(key=lambda t: severity_order.get(t.severity, 3))
    return tips
