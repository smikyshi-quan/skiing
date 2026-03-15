"""Viewpoint heuristic to detect side-view footage."""

from __future__ import annotations

from technique_analysis.common.contracts.models import FramePose

_L_ANKLE = 27
_R_ANKLE = 28
_VIS_THRESHOLD = 0.5
_BOTH_VISIBLE_FRACTION = 0.30


def detect_viewpoint(poses: list[FramePose]) -> str | None:
    """Return a warning string if side-view likely, else None.

    Heuristic: if both ankle landmarks are rarely simultaneously visible
    (both >= _VIS_THRESHOLD) in fewer than _BOTH_VISIBLE_FRACTION of frames,
    the footage is likely side-view.
    """
    if not poses:
        return None

    both_visible = 0
    for p in poses:
        lms = p.landmarks
        if len(lms) > max(_L_ANKLE, _R_ANKLE):
            if (
                lms[_L_ANKLE].visibility >= _VIS_THRESHOLD
                and lms[_R_ANKLE].visibility >= _VIS_THRESHOLD
            ):
                both_visible += 1

    fraction = both_visible / len(poses)
    if fraction < _BOTH_VISIBLE_FRACTION:
        return (
            f"Side-view likely detected: both ankles simultaneously visible in only "
            f"{fraction:.0%} of frames (threshold: {_BOTH_VISIBLE_FRACTION:.0%}). "
            "Metrics assume front-view; results may be inaccurate."
        )
    return None
