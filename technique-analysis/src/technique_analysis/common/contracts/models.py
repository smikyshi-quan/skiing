"""Session-local contracts for the technique-analysis session."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses and paths into JSON-safe structures."""
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class TechniqueRunConfig:
    pose_engine: str = "mediapipe"
    max_fps: float | None = None
    max_dimension: int | None = None  # None = auto-detect from video resolution
    render_overlay: bool = True
    render_max_dimension: int | None = None
    person_selector: str = "largest"
    min_visibility: float = 0.5
    write_debug: bool = False
    view: str = "front"
    # Bbox overlay: hidden by default (skeleton is sufficient; bbox coords are
    # in analysis-resolution space and look wrong on native-res output).
    # Pass show_bbox=True or --debug-bbox to re-enable for tracker diagnosis.
    show_bbox: bool = False

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    duration_s: float
    frame_count: int

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class FramePose:
    frame_idx: int
    timestamp_s: float
    landmarks: list[PoseLandmark]
    pose_confidence: float
    is_smoothed: bool
    # 3D world landmarks in metric space (meters, hip-centered, y=down, x=right, z=depth)
    world_landmarks: list[PoseLandmark] | None = None
    # 1.0 = fresh measurement, <1.0 = Kalman-smoothed / reduced confidence
    tracking_quality: float = 1.0
    # YOLO detection bbox in the ORIGINAL full frame (before crop/padding), pixel coords
    detection_bbox: tuple[int, int, int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class FrameMetrics:
    frame_idx: int
    timestamp_s: float
    pose_confidence: float
    knee_flexion_L: float | None
    knee_flexion_R: float | None
    hip_angle_L: float | None
    hip_angle_R: float | None
    shoulder_tilt: float | None
    hip_tilt: float | None
    knee_flexion_diff: float | None
    hip_height_diff: float | None
    stance_width_ratio: float | None
    upper_body_quietness: float | None
    hip_knee_ankle_alignment_L: float | None
    hip_knee_ankle_alignment_R: float | None
    # 3D metrics (require world landmarks; None when unavailable)
    lean_angle_deg: float | None = None   # full-body lateral lean from vertical (°)
    edge_angle_deg: float | None = None   # lower-leg inclination proxy for ski edge angle (°)
    com_shift_x: float | None = None      # lateral CoM shift (ankle_mid.x in world metres)
    com_shift_3d: float | None = None     # total CoM shift in x-z plane (metres)
    knee_angle_3d: float | None = None    # average 3D knee angle (°, world landmarks)
    # Composite scores (computed in post-processing pass)
    overall_score: float | None = None    # 0–100 weighted technique score
    movement_quality: str | None = None   # human label: Excellent / Good / Fair / Poor-…

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class TurnSummary:
    turn_idx: int
    side: str
    start_s: float
    end_s: float
    duration_s: float
    n_frames_used: int
    avg_pose_confidence: float
    avg_knee_flexion_L: float | None
    avg_knee_flexion_R: float | None
    avg_knee_flexion_diff: float | None
    avg_stance_width_ratio: float | None
    avg_upper_body_quietness: float | None
    avg_lean_angle: float | None = None
    avg_edge_angle: float | None = None
    avg_com_shift_3d: float | None = None
    # Per-turn quality (computed post-segmentation)
    quality_score: float | None = None    # 0–100 composite turn quality
    smoothness_score: float | None = None # 0–100 motion smoothness
    peak_lateral_shift: float | None = None  # max abs com_shift_x in metres
    amplitude: float | None = None        # com_shift_x peak-to-peak in metres
    # Which tracking segment this turn belongs to (0 = only/first segment)
    segment_idx: int = 0

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class CoachingTip:
    title: str
    explanation: str
    evidence: str
    severity: str
    time_ranges: list[tuple[float, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class TrackingSegment:
    """One continuous tracking epoch — a period where the tracker was locked
    to the same athlete (or a presumed-same re-lock after a brief gap).

    Multiple segments in a video indicate that either multiple athletes ran
    through the frame, or the tracker lost and re-acquired a subject after
    a significant gap.  The segment with the most high-confidence frames is
    marked is_primary=True and is the most reliable for technique analysis.
    """
    idx: int
    start_s: float
    end_s: float
    n_confident_frames: int   # frames with pose_confidence >= 0.4
    mean_confidence: float
    n_turns: int
    is_primary: bool

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class QualityReport:
    overall_pose_confidence_mean: float
    overall_pose_confidence_min: float
    low_confidence_fraction: float
    viewpoint_warning: str | None
    jitter_score_mean: float
    warnings: list[str]
    resolved_max_fps: float | None = None
    resolved_max_dimension: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class TechniqueRunSummary:
    run_id: str
    created_at: str
    video_path: str
    run_directory: str
    config: TechniqueRunConfig
    video_metadata: VideoMetadata
    quality: QualityReport
    turns: list[TurnSummary]
    coaching_tips: list[CoachingTip]
    artifacts: list[dict]
    codec_used: str
    # Tracking segments — one entry per detected athlete epoch.
    # Empty list means the whole video is treated as one segment (single athlete).
    segments: list[TrackingSegment] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["config"] = self.config.as_dict()
        payload["video_metadata"] = self.video_metadata.as_dict()
        payload["quality"] = self.quality.as_dict()
        payload["turns"] = [t.as_dict() for t in self.turns]
        payload["coaching_tips"] = [c.as_dict() for c in self.coaching_tips]
        payload["segments"] = [s.as_dict() for s in self.segments]
        return payload
