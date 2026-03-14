"""End-to-end technique analysis runner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dataclasses import replace as dc_replace

from technique_analysis.common.contracts.models import (
    FrameMetrics,
    FramePose,
    QualityReport,
    TechniqueRunConfig,
    TechniqueRunSummary,
    TrackingSegment,
)
from technique_analysis.common.coaching.rules import generate_coaching_tips
from technique_analysis.common.datasets.csv_writer import write_metrics_csv
from technique_analysis.common.datasets.paths import (
    RunPaths,
    create_run_paths,
    get_session_paths,
)
from technique_analysis.common.datasets.video_io import iter_frames, probe_video, recommend_config
from technique_analysis.common.metrics.frame_metrics import (
    compute_frame_metrics,
    compute_upper_body_quietness,
)
from technique_analysis.common.metrics.scoring import (
    compute_frame_score,
    compute_turn_quality,
)
from technique_analysis.common.pose.extractor import PoseExtractor
from technique_analysis.common.pose.vision_extractor import VisionPoseExtractor
from technique_analysis.common.pose.smoother import (
    LandmarkSmoother,
    compute_jitter_score,
)
from technique_analysis.common.pose.viewpoint import detect_viewpoint
from technique_analysis.common.rendering.overlay import render_overlay_video
from technique_analysis.common.turns.segmenter import segment_turns


def _apply_frame_scores(metrics_list: list[FrameMetrics]) -> list[FrameMetrics]:
    """Post-processing pass: fill overall_score + movement_quality on each frame."""
    result = []
    for m in metrics_list:
        score, quality = compute_frame_score(m)
        result.append(dc_replace(m, overall_score=score, movement_quality=quality))
    return result


def _apply_turn_scores(
    turns: list,
    metrics_list: list[FrameMetrics],
) -> list:
    """Post-segmentation pass: fill quality_score, smoothness_score, peak_lateral_shift."""
    result = []
    for turn in turns:
        in_turn = [
            m for m in metrics_list
            if turn.start_s <= m.timestamp_s <= turn.end_s
            and m.pose_confidence >= 0.4
        ]
        q, smooth, peak = compute_turn_quality(turn, in_turn)
        com_vals = [m.com_shift_x for m in in_turn if m.com_shift_x is not None]
        amplitude = float(max(com_vals) - min(com_vals)) if len(com_vals) >= 2 else None
        result.append(dc_replace(
            turn,
            quality_score=q,
            smoothness_score=smooth,
            peak_lateral_shift=peak,
            amplitude=amplitude,
        ))
    return result


def _fill_pose_gaps(
    poses: list[FramePose | None],
    frame_timestamps: list[tuple[int, float]],
    max_gap: int = 3,
) -> list[FramePose | None]:
    """Fill short detection gaps by carrying the last valid pose forward.

    Each filled frame gets a confidence and tracking_quality that decay
    linearly with gap distance so metrics stay conservative.
    """
    result: list[FramePose | None] = list(poses)
    last_valid_i = -1
    for i, pose in enumerate(result):
        if pose is not None:
            last_valid_i = i
        elif last_valid_i >= 0:
            gap = i - last_valid_i
            if gap <= max_gap:
                src = result[last_valid_i]
                decay = max(0.1, 1.0 - gap * 0.25)
                fidx, ts = frame_timestamps[i]
                result[i] = FramePose(
                    frame_idx=fidx,
                    timestamp_s=ts,
                    landmarks=src.landmarks,
                    world_landmarks=src.world_landmarks,
                    pose_confidence=src.pose_confidence * decay,
                    is_smoothed=src.is_smoothed,
                    tracking_quality=decay,
                )
    return result


def _estimate_median_torso_size(poses: list[FramePose | None]) -> float:
    """Median shoulder-to-hip vertical distance across valid poses."""
    sizes = []
    for p in poses:
        if p is None or len(p.landmarks) < 25:
            continue
        lms = p.landmarks
        mid_shoulder_y = (lms[11].y + lms[12].y) / 2
        mid_hip_y = (lms[23].y + lms[24].y) / 2
        sizes.append(abs(mid_hip_y - mid_shoulder_y))
    return float(np.median(sizes)) if sizes else 0.15


def _empty_metrics(frame_idx: int, timestamp_s: float) -> FrameMetrics:
    return FrameMetrics(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        pose_confidence=0.0,
        knee_flexion_L=None,
        knee_flexion_R=None,
        hip_angle_L=None,
        hip_angle_R=None,
        shoulder_tilt=None,
        hip_tilt=None,
        knee_flexion_diff=None,
        hip_height_diff=None,
        stance_width_ratio=None,
        upper_body_quietness=None,
        hip_knee_ankle_alignment_L=None,
        hip_knee_ankle_alignment_R=None,
    )


def _build_quality_report(
    metrics_list: list[FrameMetrics],
    all_poses: list[FramePose | None],
    viewpoint_warning: str | None,
    resolved_max_fps: float | None,
    resolved_max_dimension: int,
) -> QualityReport:
    if not metrics_list:
        return QualityReport(
            overall_pose_confidence_mean=0.0,
            overall_pose_confidence_min=0.0,
            low_confidence_fraction=1.0,
            viewpoint_warning=viewpoint_warning,
            jitter_score_mean=0.0,
            warnings=[],
            resolved_max_fps=resolved_max_fps,
            resolved_max_dimension=resolved_max_dimension,
        )
    confs = [m.pose_confidence for m in metrics_list]
    low_conf_thresh = 0.4
    low_frac = sum(1 for c in confs if c < low_conf_thresh) / len(confs)
    valid_poses = [p for p in all_poses if p is not None]
    jitter = compute_jitter_score(valid_poses)
    warnings: list[str] = []
    if viewpoint_warning:
        warnings.append(viewpoint_warning)
    if low_frac > 0.5:
        warnings.append(f"High low-confidence frame fraction: {low_frac:.0%}")
    return QualityReport(
        overall_pose_confidence_mean=float(np.mean(confs)),
        overall_pose_confidence_min=float(np.min(confs)),
        low_confidence_fraction=low_frac,
        viewpoint_warning=viewpoint_warning,
        jitter_score_mean=jitter,
        warnings=warnings,
        resolved_max_fps=resolved_max_fps,
        resolved_max_dimension=resolved_max_dimension,
    )


def _build_segments(
    segment_boundaries: list[float],
    metrics_list: list[FrameMetrics],
    turns: list,
    video_duration_s: float,
) -> tuple[list[TrackingSegment], list]:
    """Partition metrics and turns into tracking segments.

    Each segment covers the period between two consecutive boundary timestamps.
    The segment with the most high-confidence frames is marked is_primary=True.
    Turns are annotated with the segment_idx they belong to.
    """
    ends = segment_boundaries[1:] + [video_duration_s]

    segments: list[TrackingSegment] = []
    for idx, (start_s, end_s) in enumerate(zip(segment_boundaries, ends)):
        seg_metrics = [m for m in metrics_list if start_s <= m.timestamp_s < end_s]
        confident   = [m for m in seg_metrics if m.pose_confidence >= 0.4]
        mean_conf   = float(np.mean([m.pose_confidence for m in confident])) if confident else 0.0
        n_turns     = sum(
            1 for t in turns if t.start_s >= start_s and t.end_s <= end_s
        )
        segments.append(TrackingSegment(
            idx=idx,
            start_s=round(start_s, 2),
            end_s=round(end_s, 2),
            n_confident_frames=len(confident),
            mean_confidence=round(mean_conf, 3),
            n_turns=n_turns,
            is_primary=False,
        ))

    # Mark the segment with the most high-confidence frames as primary
    if segments:
        best = max(range(len(segments)), key=lambda i: segments[i].n_confident_frames)
        segments[best] = dc_replace(segments[best], is_primary=True)

    # Annotate each turn with its segment_idx
    annotated: list = []
    for turn in turns:
        seg_idx = 0
        for seg in segments:
            if turn.start_s >= seg.start_s and turn.end_s <= seg.end_s:
                seg_idx = seg.idx
                break
        annotated.append(dc_replace(turn, segment_idx=seg_idx))

    return segments, annotated


class TechniqueAnalysisRunner:
    """Run the local-only technique analysis pipeline."""

    def __init__(self, config: TechniqueRunConfig | None = None) -> None:
        self.config = config or TechniqueRunConfig()

    def run(self, video_path: str | Path) -> TechniqueRunSummary:
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        session_paths = get_session_paths()
        session_paths.ensure()
        run_paths = create_run_paths(video_path)
        run_paths.ensure()

        # 1. Probe video
        metadata = probe_video(video_path)

        # Resolve auto settings based on video resolution/FPS
        auto_fps, auto_dim = recommend_config(metadata)
        resolved_max_fps = self.config.max_fps if self.config.max_fps is not None else auto_fps
        resolved_max_dimension = self.config.max_dimension if self.config.max_dimension is not None else auto_dim
        print(
            f"  Video: {metadata.width}×{metadata.height} @ {metadata.fps:.1f} fps  "
            f"→ analysis: max_dimension={resolved_max_dimension}, max_fps={resolved_max_fps}"
        )

        # 2. Extract poses (streaming)
        #
        # Key design: YOLO ByteTrack runs on EVERY native frame so it sees
        # continuous motion and maintains accurate track IDs even for fast-
        # moving skiers.  MediaPipe pose estimation only runs on analysis
        # frames (at resolved_max_fps) to keep the pipeline fast.
        all_poses: list[FramePose | None] = []
        smoother = LandmarkSmoother()
        frame_timestamps: list[tuple[int, float]] = []

        analysis_interval_s = (
            1.0 / resolved_max_fps if resolved_max_fps is not None else 0.0
        )
        next_analysis_ts = 0.0

        ExtractorClass = (
            VisionPoseExtractor
            if self.config.pose_engine == "vision"
            else PoseExtractor
        )
        with ExtractorClass(min_visibility=self.config.min_visibility) as extractor:
            for frame_idx, timestamp_s, frame in iter_frames(
                video_path,
                max_fps=None,               # read EVERY frame for ByteTrack
                max_dimension=resolved_max_dimension,
            ):
                is_analysis = (
                    resolved_max_fps is None
                    or timestamp_s >= next_analysis_ts - 1e-4
                )
                if is_analysis:
                    frame_timestamps.append((frame_idx, timestamp_s))
                    pose = extractor.extract(frame, frame_idx, timestamp_s)
                    if pose is not None:
                        pose = smoother.smooth(pose)
                    all_poses.append(pose)
                    next_analysis_ts = timestamp_s + analysis_interval_s
                else:
                    extractor.update_tracking(frame)  # ByteTrack only

        scene_cuts = extractor.scene_cuts_detected

        # 3. Fill detection gaps — 8 frames covers the typical 0.3-0.4s
        #    carve-transition window where MediaPipe confidence drops
        all_poses = _fill_pose_gaps(all_poses, frame_timestamps, max_gap=8)

        # 4. Viewpoint heuristic
        valid_poses = [p for p in all_poses if p is not None]
        viewpoint_warning = detect_viewpoint(valid_poses)

        # 5. Compute per-frame metrics
        torso_size = _estimate_median_torso_size(all_poses)
        raw_metrics: list[FrameMetrics] = []
        for pose, (frame_idx, timestamp_s) in zip(all_poses, frame_timestamps):
            if pose is not None:
                raw_metrics.append(compute_frame_metrics(pose, torso_size))
            else:
                raw_metrics.append(_empty_metrics(frame_idx, timestamp_s))

        # Second pass: fill upper_body_quietness
        metrics_list = compute_upper_body_quietness(raw_metrics, all_poses)

        # Third pass: fill overall_score + movement_quality
        metrics_list = _apply_frame_scores(metrics_list)

        # 5. Segment turns
        turns = segment_turns(metrics_list)
        turns = _apply_turn_scores(turns, metrics_list)

        # 5b. Build tracking segments and annotate turns with segment_idx
        segments, turns = _build_segments(
            extractor.segment_boundaries,
            metrics_list,
            turns,
            metadata.duration_s,
        )

        # 6. Quality report
        quality = _build_quality_report(
            metrics_list, all_poses, viewpoint_warning,
            resolved_max_fps, resolved_max_dimension,
        )
        if scene_cuts > 0:
            quality.warnings.append(
                f"{scene_cuts} scene cut(s) detected — metrics may span multiple shots. "
                "Trim to a single continuous run for best results."
            )
        if len(segments) > 1:
            primary = next(s for s in segments if s.is_primary)
            quality.warnings.append(
                f"{len(segments)} tracking segments detected — multiple athletes likely. "
                f"Primary segment: t={primary.start_s:.1f}–{primary.end_s:.1f}s "
                f"({primary.n_confident_frames} high-confidence frames, "
                f"{primary.n_turns} turns). "
                "Trim to a single continuous run for accurate analysis."
            )

        # 7. Coaching tips
        coaching_tips = generate_coaching_tips(metrics_list, turns, quality)

        # 8. Write metrics CSV
        write_metrics_csv(metrics_list, run_paths.metrics_csv_path)

        # 9. Render overlay
        codec = "none"
        if self.config.render_overlay:
            try:
                codec = render_overlay_video(
                    video_path=video_path,
                    poses=all_poses,
                    metrics_list=metrics_list,
                    turns=turns,
                    output_path=run_paths.overlay_path,
                    max_dimension=self.config.render_max_dimension,
                    show_bbox=self.config.show_bbox,
                )
            except Exception as exc:
                quality.warnings.append(f"Overlay rendering failed: {exc}")

        # 10. Write summary JSON
        summary = TechniqueRunSummary(
            run_id=run_paths.run_id,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            video_path=str(video_path),
            run_directory=str(run_paths.run_dir),
            config=self.config,
            video_metadata=metadata,
            quality=quality,
            turns=turns,
            coaching_tips=coaching_tips,
            artifacts=[
                {"kind": "video_overlay", "path": str(run_paths.overlay_path)},
                {"kind": "metrics_csv", "path": str(run_paths.metrics_csv_path)},
                {"kind": "summary_json", "path": str(run_paths.summary_json_path)},
            ],
            codec_used=codec,
            segments=segments,
        )
        run_paths.summary_json_path.write_text(
            json.dumps(summary.as_dict(), indent=2), encoding="utf-8"
        )
        return summary
