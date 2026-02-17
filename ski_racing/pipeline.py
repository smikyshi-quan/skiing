"""
End-to-end ski racing analysis pipeline.
Video in -> gates + 3D trajectory + physics validation out.

Supports camera stabilization mode (--stabilize) which enables:
  Phase 1: Per-frame gate detection and tracking
  Phase 2: Camera motion compensation from gate anchors
  Phase 3: Kalman filter trajectory smoothing
  Phase 4: Dynamic per-frame pixel-to-meter scaling

Usage:
    from ski_racing.pipeline import SkiRacingPipeline
    pipeline = SkiRacingPipeline("models/gate_detector_best.pt")
    results = pipeline.process_video("race.mp4")

    # With stabilization:
    pipeline = SkiRacingPipeline("models/gate_detector_best.pt", stabilize=True)
    results = pipeline.process_video("race.mp4")
"""
import cv2
import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from .detection import GateDetector, TemporalGateTracker
from .tracking import SkierTracker, KalmanSmoother, TrajectoryOutlierFilter
from .transform import HomographyTransform, CameraMotionCompensator, DynamicScaleTransform
from .physics import PhysicsValidator


class SkiRacingPipeline:
    """
    End-to-end pipeline: video -> gates + 3D trajectory + physics validation.
    """

    SUPPORTED_DISCIPLINES = ("slalom", "giant_slalom")
    DEFAULT_GATE_SPACING_BY_DISCIPLINE = {
        "slalom": 9.5,
        "giant_slalom": 27.0,
    }

    def __init__(self, gate_model_path, discipline=None, gate_spacing_m=None,
                 stabilize=False, camera_mode="affine", camera_pitch_deg=None):
        """
        Args:
            gate_model_path: Path to trained YOLOv8 gate detection weights.
            discipline: "slalom" or "giant_slalom". If None, auto-detected
                        from gate geometry per video.
            gate_spacing_m: Expected distance between gates in meters.
                           If None, auto-uses discipline defaults:
                           slalom=9.5m, giant_slalom=27m.
            stabilize: Enable all 4 improvement phases (camera compensation,
                      Kalman smoothing, dynamic scale, per-frame gates).
            camera_mode: "affine" (default) or "translation" for camera motion.
            camera_pitch_deg: Camera pitch angle in degrees for scale correction.
        """
        self.gate_detector = GateDetector(gate_model_path)
        self.skier_tracker = SkierTracker()
        self.transformer = HomographyTransform()
        if discipline is not None and discipline not in self.SUPPORTED_DISCIPLINES:
            raise ValueError(
                f"Unsupported discipline '{discipline}'. "
                f"Supported: {', '.join(self.SUPPORTED_DISCIPLINES)}"
            )
        self.discipline = discipline
        self._discipline_explicit = discipline is not None
        self.discipline_source = "explicit" if self._discipline_explicit else "auto_detected"

        self._gate_spacing_explicit = gate_spacing_m is not None
        if self._gate_spacing_explicit:
            self.gate_spacing_m = float(gate_spacing_m)
            self.gate_spacing_source = "explicit"
        else:
            self.gate_spacing_m = None
            self.gate_spacing_source = "discipline_default_pending"
        self.stabilize = stabilize
        self.camera_mode = camera_mode
        self.camera_pitch_deg = camera_pitch_deg

    @staticmethod
    def classify_discipline(gates, frame_height):
        """
        Classify discipline from detected gates in the best frame.

        Rules:
          - >=6 visible gates: slalom
          - <=3 visible gates: giant_slalom
          - 4-5 gates: use median consecutive Y-gap
              gap < 15% of frame height -> slalom
              else -> giant_slalom
        """
        gates = gates or []
        gate_count = len(gates)
        sorted_gates = sorted(gates, key=lambda g: float(g.get("base_y", 0.0)))
        ys = [float(g.get("base_y", 0.0)) for g in sorted_gates]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1e-6]
        median_gap_px = float(np.median(gaps)) if gaps else None
        gap_ratio = None
        if median_gap_px is not None and frame_height and frame_height > 0:
            gap_ratio = float(median_gap_px / frame_height)

        if gate_count >= 6:
            discipline = "slalom"
            rule = "gate_count>=6"
        elif gate_count <= 3:
            discipline = "giant_slalom"
            rule = "gate_count<=3"
        else:
            threshold_ratio = 0.15
            if gap_ratio is not None and gap_ratio < threshold_ratio:
                discipline = "slalom"
                rule = "median_gap<15%_height"
            else:
                discipline = "giant_slalom"
                rule = "median_gap>=15%_height"

        return {
            "discipline": discipline,
            "gate_count": int(gate_count),
            "median_gap_px": median_gap_px,
            "median_gap_ratio": gap_ratio,
            "rule": rule,
        }

    def process_video(
        self,
        video_path,
        output_dir="artifacts/outputs",
        validate_physics=True,
        projection="scale",
        gate_conf=0.35,
        gate_iou=0.55,
        skier_conf=0.25,
        gate_search_frames=150,
        gate_search_stride=5,
        gate_track_frames=120,
        gate_track_stride=3,
        gate_track_min_obs=3,
        frame_stride=1,
        max_frames=None,
        max_jump=None,
        kalman_process_noise=None,
        camera_compensate_before_smoothing=False,
        enable_dynamic_scale=True,
    ):
        """
        Full analysis pipeline.

        Args:
            video_path: Path to race video.
            output_dir: Directory for saving results.
            validate_physics: If True, run physics validation on trajectory.

        Returns:
            Dictionary with all analysis results.
        """
        video_path = str(video_path)
        Path(output_dir).mkdir(exist_ok=True)

        # Phase 1 (Prof. feedback): stride>1 introduces micro-drifts in
        # camera stabilization — force stride=1 for production quality.
        if self.stabilize and gate_track_stride != 1:
            print(f"⚠️  Stabilize mode forces gate_track_stride=1 (was {gate_track_stride})")
            gate_track_stride = 1
        elif not self.stabilize and gate_track_stride > 1:
            print(f"⚠️  gate_track_stride={gate_track_stride}: skipping frames can introduce "
                  f"micro-drifts in gate positions (~{gate_track_stride * 0.033:.2f}s gaps). "
                  f"Use stride=1 for production accuracy.")

        # Phase 4 (Prof. feedback): Static homography calculated once is WRONG
        # because perspective changes as skier moves down the slope. Dynamic
        # scale is more stable than a noisy global homography.
        if self.stabilize and projection == "homography":
            print("⚠️  Stabilize mode uses dynamic scale; forcing projection='scale'")
            projection = "scale"
        elif projection == "homography":
            print("⚠️  WARNING: Static homography is calculated once and applied to "
                  "the whole video. Perspective changes as the skier moves down "
                  "the slope. Consider using --projection scale or --stabilize "
                  "for dynamic per-frame scaling.")

        # CRITICAL SAFETY: Affine + Dynamic Scale is a dangerous combination.
        # Affine can hallucinate zoom in low-texture snow scenes, which feeds
        # into DynamicScale and creates explosive meters-per-pixel values.
        # The affine mode now strips its scale component (Euclidean only),
        # but translation mode is still safer for most footage.
        if self.stabilize and self.camera_mode == "affine":
            print("⚠️  Affine mode + dynamic scale: the affine transform's scale "
                  "component has been stripped (Euclidean only = rotation+translation). "
                  "If results are still unstable, switch to --camera-mode translation.")

        # Get video info
        video_info = self.skier_tracker.get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]

        mode_label = " [STABILIZED]" if self.stabilize else ""
        print(f"Processing {Path(video_path).name}...{mode_label}")
        print(f"  Video: {video_info['width']}x{video_info['height']} @ {fps:.1f} fps, "
              f"{total_frames} frames")

        n_steps = 7 if self.stabilize else 5
        phase_timings = {
            "gate_detection_initial_frame_search": 0.0,
            "gate_tracking_full_video_pass": 0.0,
            "perspective_transform_calculation": 0.0,
            "camera_motion_compensation": 0.0,
            "skier_tracking_full_video_pass": 0.0,
            "kalman_smoothing": 0.0,
            "trajectory_3d_transform": 0.0,
            "physics_validation": 0.0,
        }

        # ─── Step 1: Detect gates from first/best frame ───
        phase_t0 = time.time()
        print(f"  [1/{n_steps}] Detecting gates...")
        gates = self.gate_detector.detect_from_first_frame(
            video_path,
            conf=gate_conf,
            iou=gate_iou,
        )
        if len(gates) < 4 and gate_search_frames > 0:
            print("         Low gate count; scanning additional frames...")
            gates = self.gate_detector.detect_from_best_frame(
                video_path,
                conf=gate_conf,
                iou=gate_iou,
                max_frames=gate_search_frames,
                stride=gate_search_stride,
            )

        # Early dedup to avoid duplicate poles corrupting tracking/scale
        frame_height = video_info["height"]
        if len(gates) >= 2:
            gates = self._cluster_gates_by_y(gates, y_thresh=15.0, frame_height=frame_height)
        gates_initial = [dict(g) for g in gates]
        phase_timings["gate_detection_initial_frame_search"] = float(time.time() - phase_t0)

        # ─── Discipline and default spacing resolution ───
        discipline_info = None
        if not self._discipline_explicit:
            discipline_info = self.classify_discipline(gates_initial, video_info["height"])
            self.discipline = discipline_info["discipline"]
            self.discipline_source = "auto_detected"
            gate_count = discipline_info["gate_count"]
            median_gap_px = discipline_info["median_gap_px"]
            gap_ratio = discipline_info["median_gap_ratio"]
            if median_gap_px is not None and gap_ratio is not None:
                print(f"         Auto-detected: {self.discipline} "
                      f"({gate_count} gates visible, median gap {median_gap_px:.0f}px "
                      f"= {gap_ratio * 100:.1f}% of frame height)")
            else:
                print(f"         Auto-detected: {self.discipline} ({gate_count} gates visible)")
        else:
            self.discipline_source = "explicit"

        if not self._gate_spacing_explicit:
            self.gate_spacing_m = float(self.DEFAULT_GATE_SPACING_BY_DISCIPLINE[self.discipline])
            self.gate_spacing_source = "discipline_default"
            print(f"         Using default gate spacing for {self.discipline}: {self.gate_spacing_m:.1f}m")
        else:
            self.gate_spacing_source = "explicit"

        # ─── Step 1b: Per-frame gate tracking (Phase 1) ───
        frame_gate_history = None
        frame_gate_history_full = None
        baseline_gates = None

        phase_t0 = time.time()
        if self.stabilize and len(gates) >= 2:
            print(f"  [2/{n_steps}] Tracking gates across all frames (Phase 1)...")
            gates, frame_gate_history, frame_gate_history_full, baseline_gates = self._track_gates_full_video(
                video_path,
                gates,
                conf=gate_conf,
                iou=gate_iou,
                stride=gate_track_stride,
            )
            phase_timings["gate_tracking_full_video_pass"] = float(time.time() - phase_t0)
        elif gate_track_frames > 0 and len(gates) >= 2:
            print("         Stabilizing gates with temporal tracking...")
            gates = self._refine_gates_with_tracking(
                video_path,
                gates,
                conf=gate_conf,
                iou=gate_iou,
                max_frames=gate_track_frames,
                stride=gate_track_stride,
                min_obs=gate_track_min_obs,
            )
            phase_timings["gate_tracking_full_video_pass"] = float(time.time() - phase_t0)

        if len(gates) >= 2:
            gates = self._cluster_gates_by_y(gates, y_thresh=15.0, frame_height=frame_height)
        print(f"         Found {len(gates)} gates")

        # Optional auto-estimation if pitch wasn't provided by caller.
        camera_pitch_deg = self.camera_pitch_deg
        if camera_pitch_deg is None and frame_gate_history:
            estimated_pitch = DynamicScaleTransform.estimate_camera_pitch_deg_from_history(frame_gate_history)
            if estimated_pitch is not None:
                camera_pitch_deg = float(estimated_pitch)
                print(f"         Estimated camera pitch from gate geometry: {camera_pitch_deg:.1f}°")

        # ─── Step 2/3: Calculate projection transform ───
        step_n = 3 if self.stabilize else 2
        phase_t0 = time.time()
        print(f"  [{step_n}/{n_steps}] Calculating perspective transform...")
        # Use raw (unstabilized) gate positions for scale mapping when stabilizing
        scale_gates = gates_initial if (self.stabilize and len(gates_initial) >= 2) else gates
        gate_centers = [[g["center_x"], g["base_y"]] for g in scale_gates]
        self.transformer.camera_pitch_deg = camera_pitch_deg
        if projection == "scale":
            self.transformer.calculate_scale_from_gates(gate_centers, self.gate_spacing_m)
        else:
            self.transformer.calculate_from_gates(gate_centers, self.gate_spacing_m)
        phase_timings["perspective_transform_calculation"] = float(time.time() - phase_t0)

        # ─── Phase 2: Camera motion compensation ───
        if self.stabilize and frame_gate_history and baseline_gates:
            phase_t0 = time.time()
            print(f"  [{step_n}/{n_steps}]   + Camera motion compensation (Phase 2)...")
            compensator = CameraMotionCompensator(
                baseline_gates,
                frame_gate_history,
                mode=self.camera_mode,
            )
            compensator.estimate_motion()
            self.transformer.camera_compensator = compensator
            phase_timings["camera_motion_compensation"] = float(time.time() - phase_t0)

        # ─── Phase 4: Dynamic scale ───
        if self.stabilize and frame_gate_history and enable_dynamic_scale:
            print(f"  [{step_n}/{n_steps}]   + Dynamic per-frame scale (Phase 4)...")
            dyn_scale = DynamicScaleTransform(
                frame_gate_history,
                self.gate_spacing_m,
                camera_pitch_deg=camera_pitch_deg,
            )
            one_gate_ref = None
            if self.transformer.y_map and len(self.transformer.y_map) > 1:
                # Use the same one-gate reference from the static scale
                gaps = []
                for i in range(1, len(self.transformer.y_map)):
                    dy = self.transformer.y_map[i]["y_px"] - self.transformer.y_map[i - 1]["y_px"]
                    if dy > 1e-3:
                        gaps.append(dy)
                if gaps:
                    gaps_sorted = sorted(gaps)
                    one_gate_ref = gaps_sorted[max(0, int(0.2 * (len(gaps_sorted) - 1)))]
            dyn_scale.compute_scales(self.transformer.ppm_y, one_gate_ref)
            self.transformer.dynamic_scale = dyn_scale
        elif self.stabilize and frame_gate_history and not enable_dynamic_scale:
            self.transformer.dynamic_scale = None
            print(f"  [{step_n}/{n_steps}]   + Dynamic per-frame scale (Phase 4) DISABLED")

        # ─── Step 3/4: Track skier ───
        step_n = 4 if self.stabilize else 3
        phase_t0 = time.time()
        print(f"  [{step_n}/{n_steps}] Tracking skier...")
        trajectory_2d = self.skier_tracker.track_video(
            video_path,
            method="bytetrack",
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_jump=max_jump,
            conf=skier_conf,
        )
        phase_timings["skier_tracking_full_video_pass"] = float(time.time() - phase_t0)
        tracking_diag = self.skier_tracker.get_last_tracking_stats()
        bytetrack_cov = float(tracking_diag.get("bytetrack_coverage", 0.0))
        track_switches = int(tracking_diag.get("track_id_switches", 0))
        print(f"         Tracked {len(trajectory_2d)} positions "
              f"({len(trajectory_2d)/total_frames*100:.0f}% coverage)")
        if tracking_diag.get("method_requested") == "bytetrack":
            print(f"         ByteTrack coverage: {bytetrack_cov:.1%}, "
                  f"ID switches: {track_switches}")

        # Keep raw trajectory for comparison
        trajectory_2d_raw = None
        trajectory_2d_original = [dict(pt) for pt in trajectory_2d]
        outlier_info = {"outlier_count": 0, "outlier_frames": []}

        # Optional ordering experiment:
        # track -> camera compensate 2D -> outlier filter -> smooth
        if (
            self.stabilize
            and camera_compensate_before_smoothing
            and self.transformer.camera_compensator is not None
            and len(trajectory_2d) > 0
        ):
            print(f"  [{step_n}/{n_steps}] Applying 2D camera compensation before smoothing (experiment)...")
            compensated = []
            for pt in trajectory_2d:
                x_stab, y_stab = self.transformer.camera_compensator.stabilize_point(
                    pt["x"], pt["y"], pt.get("frame")
                )
                upd = dict(pt)
                upd["x"] = float(x_stab)
                upd["y"] = float(y_stab)
                compensated.append(upd)
            trajectory_2d = compensated

        # Pre-smoothing outlier rejection
        if self.stabilize and len(trajectory_2d) > 0:
            outlier_filter = TrajectoryOutlierFilter(window=5, mad_threshold=3.0)
            trajectory_2d, outlier_info = outlier_filter.filter(trajectory_2d)
            outlier_count = int(outlier_info.get("outlier_count", 0))
            if outlier_count > 0:
                print(f"         Outlier filter: corrected {outlier_count} frames")
            if total_frames > 0 and outlier_count > 0.05 * total_frames:
                print(f"  ⚠️  Tracking quality warning: {outlier_count}/{total_frames} "
                      f"frames flagged as outliers (>5%).")

        # ─── Phase 3: Kalman smoothing ───
        if self.stabilize and len(trajectory_2d) > 2:
            step_n = 5
            phase_t0 = time.time()
            print(f"  [{step_n}/{n_steps}] Smoothing trajectory with Kalman filter (Phase 3)...")
            trajectory_2d_raw = [dict(pt) for pt in trajectory_2d]  # Deep copy
            kf = KalmanSmoother(
                fps=fps,
                discipline=self.discipline,
                process_noise=kalman_process_noise,
            )
            trajectory_2d = kf.smooth(trajectory_2d)
            phase_timings["kalman_smoothing"] = float(time.time() - phase_t0)

        # ─── Step 4/6: Transform to 3D ───
        step_n = 6 if self.stabilize else 4
        phase_t0 = time.time()
        print(f"  [{step_n}/{n_steps}] Transforming to 3D coordinates...")
        apply_transform_stabilization = bool(self.stabilize and not camera_compensate_before_smoothing)
        trajectory_3d = self.transformer.transform_trajectory(
            trajectory_2d,
            stabilize=apply_transform_stabilization,
            fps=fps,
            stabilize_after_scale=True,
        )

        # ── Guard against singularities / extreme jumps before calibration ──
        trajectory_3d, sanitize_info = self._sanitize_trajectory_3d(trajectory_3d, fps)

        # ── Speed-based scale auto-calibration ──
        # Gate-based ppm estimation is fragile: the detector may find every
        # other gate, or double-detect single gates, or struggle with
        # perspective foreshortening. If the resulting P90 speed is
        # unrealistic for the discipline, we compute a correction factor.
        trajectory_3d, auto_calibration = self._auto_calibrate_scale(
            trajectory_3d, fps, trajectory_2d,
        )

        # ── Optional 3D smoothing before physics ──
        trajectory_3d_raw = [dict(pt) for pt in trajectory_3d]
        trajectory_3d = self._smooth_trajectory_3d(trajectory_3d, window=5)
        phase_timings["trajectory_3d_transform"] = float(time.time() - phase_t0)

        # ─── Step 5/7: Physics validation ───
        step_n = 7 if self.stabilize else 5
        phase_t0 = time.time()
        physics_result = None
        if validate_physics and len(trajectory_3d) > 3 and len(gates) >= 2:
            print(f"  [{step_n}/{n_steps}] Running physics validation...")
            validator = PhysicsValidator(discipline=self.discipline, fps=fps)
            physics_result = validator.validate_trajectory(trajectory_3d)
            validator.print_report(physics_result)
        else:
            print(f"  [{step_n}/{n_steps}] Skipping physics validation (insufficient data)")
            if len(gates) < 2:
                physics_result = {
                    "valid": False,
                    "issues": ["Insufficient gate detections for scale/homography"],
                    "metrics": {},
                    "discipline": self.discipline,
                    "fps": fps,
                }
        phase_timings["physics_validation"] = float(time.time() - phase_t0)

        print("  Runtime profile (seconds):")
        print(f"    Gate detection (initial frame search): {phase_timings['gate_detection_initial_frame_search']:.3f}")
        print(f"    Gate tracking (full video pass): {phase_timings['gate_tracking_full_video_pass']:.3f}")
        print(f"    Perspective transform calculation: {phase_timings['perspective_transform_calculation']:.3f}")
        print(f"    Camera motion compensation: {phase_timings['camera_motion_compensation']:.3f}")
        print(f"    Skier tracking (full video pass): {phase_timings['skier_tracking_full_video_pass']:.3f}")
        print(f"    Kalman smoothing: {phase_timings['kalman_smoothing']:.3f}")
        print(f"    3D trajectory transform: {phase_timings['trajectory_3d_transform']:.3f}")
        print(f"    Physics validation: {phase_timings['physics_validation']:.3f}")

        # Compile results
        results = {
            "video": video_path,
            "video_info": video_info,
            "discipline": self.discipline,
            "discipline_source": self.discipline_source,
            "discipline_detection": discipline_info,
            "gate_spacing_m": self.gate_spacing_m,
            "gate_spacing_source": self.gate_spacing_source,
            "camera_pitch_deg": camera_pitch_deg,
            "effective_gate_spacing_m": self.transformer.effective_gate_spacing_m,
            "projection": projection,
            "gate_conf": float(gate_conf),
            "gate_iou": float(gate_iou),
            "stabilized": self.stabilize,
            "camera_mode": self.camera_mode if self.stabilize else None,
            "gates": gates,
            "trajectory_2d": trajectory_2d,
            "trajectory_3d": trajectory_3d,
            "physics_validation": physics_result,
            "timestamp": datetime.now().isoformat(),
            "auto_calibration": auto_calibration,
            "trajectory_sanitize": sanitize_info,
            "outlier_count": int(outlier_info.get("outlier_count", 0)),
            "outlier_frames": [int(f) for f in outlier_info.get("outlier_frames", [])],
            "bytetrack_coverage": float(tracking_diag.get("bytetrack_coverage", 0.0)),
            "track_id_switches": int(tracking_diag.get("track_id_switches", 0)),
            "tracking_diagnostics": tracking_diag,
            "camera_compensate_before_smoothing": bool(camera_compensate_before_smoothing),
            "kalman_process_noise": float(kalman_process_noise) if kalman_process_noise is not None else None,
            "dynamic_scale_enabled": bool(enable_dynamic_scale),
            "runtime_profile_sec": phase_timings,
        }

        # Add stabilization-specific data
        if self.stabilize:
            results["kalman_smoothed"] = True
            if trajectory_2d_raw is not None:
                results["trajectory_2d_raw"] = trajectory_2d_raw
            if trajectory_2d_original is not None:
                results["trajectory_2d_original"] = trajectory_2d_original
            if trajectory_3d_raw is not None:
                results["trajectory_3d_raw"] = trajectory_3d_raw
            if self.transformer.camera_compensator:
                if self.camera_mode == "affine":
                    results["camera_motion_frames"] = len(self.transformer.camera_compensator.affine)
                else:
                    results["camera_motion_frames"] = len(self.transformer.camera_compensator.offsets)
            if frame_gate_history:
                # Detected-only frames (used for scaling/stabilization)
                results["frames_detected"] = self._build_frame_records(frame_gate_history)
                results["gate_track_stride"] = gate_track_stride
            if frame_gate_history_full:
                # Full per-frame gates (includes interpolated/missing)
                results["frames"] = self._build_frame_records(frame_gate_history_full)
            if self.transformer.dynamic_scale is not None:
                results["dynamic_scale"] = self.transformer.dynamic_scale.to_debug_dict()

        # Save results
        output_path = Path(output_dir) / f"{Path(video_path).stem}_analysis.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Results saved to {output_path}")
        return results

    def _build_frame_records(self, frame_gate_history):
        """
        Build per-frame gate records for JSON output.
        """
        frames = []
        for frame_idx in sorted(frame_gate_history.keys()):
            gates = []
            for gate_id, payload in frame_gate_history[frame_idx].items():
                if isinstance(payload, dict):
                    gate_info = dict(payload)
                    gate_info.setdefault("gate_id", int(gate_id))
                    # Normalize numeric fields
                    if "center_x" in gate_info:
                        gate_info["center_x"] = float(gate_info["center_x"])
                    if "base_y" in gate_info:
                        gate_info["base_y"] = float(gate_info["base_y"])
                    if "confidence" in gate_info:
                        gate_info["confidence"] = float(gate_info["confidence"])
                    if "missing_frames" in gate_info:
                        gate_info["missing_frames"] = int(gate_info["missing_frames"])
                    gates.append(gate_info)
                else:
                    cx, by = payload
                    gates.append({
                        "gate_id": int(gate_id),
                        "center_x": float(cx),
                        "base_y": float(by),
                    })
            frames.append({
                "frame": int(frame_idx),
                "gates": gates,
            })
        return frames

    def _track_gates_full_video(self, video_path, initial_gates, conf=0.35, iou=0.55, stride=3):
        """
        Track gates across the entire video to build per-frame gate history.

        Returns:
            Tuple of (refined_gates, frame_gate_history, frame_gate_history_full, baseline_gates)
            - refined_gates: List of gate dicts with median positions
            - frame_gate_history: {frame_idx: {gate_id: (cx, by)}} (detected-only)
            - frame_gate_history_full: {frame_idx: {gate_id: gate_info}} (includes interpolated)
            - baseline_gates: {gate_id: (cx, by)} for camera compensation
        """
        tracker = TemporalGateTracker(max_missing_frames=30, match_threshold=60.0)
        tracker.initialize(initial_gates)

        history_for_median = {}  # gate_id -> [(cx, by, class, name)]
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        stride = max(1, int(stride))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            detections = self.gate_detector.detect_in_frame(frame, conf=conf, iou=iou)
            tracked = tracker.update_with_frame_idx(detections, frame_idx)

            for g in tracked:
                if not g.get("is_interpolated", False):
                    history_for_median.setdefault(g["gate_id"], []).append(
                        (g["center_x"], g["base_y"], g["class"],
                         g.get("class_name", "gate"), g.get("confidence", 0.0))
                    )

            processed += 1
            frame_idx += 1

        cap.release()

        # Build refined gates (median positions)
        refined = []
        baseline = {}
        for gate_id, items in history_for_median.items():
            if len(items) < 3:
                continue
            xs = [i[0] for i in items]
            ys = [i[1] for i in items]
            cls = items[0][2]
            name = items[0][3]
            confs = [i[4] for i in items if len(i) > 4]
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            med_conf = float(np.median(confs)) if confs else 0.0
            refined.append({
                "gate_id": gate_id,
                "center_x": med_x,
                "base_y": med_y,
                "class": int(cls),
                "class_name": name,
                "confidence": med_conf,
            })
            baseline[gate_id] = (med_x, med_y)

        if not refined:
            return initial_gates, {}, {}, {}

        refined.sort(key=lambda g: g["base_y"])
        frame_gate_history = tracker.get_frame_history()
        frame_gate_history_full = tracker.get_frame_history_full()

        print(f"         Tracked {len(refined)} gates across {processed} frames "
              f"({len(frame_gate_history)} frames with detections)")

        return refined, frame_gate_history, frame_gate_history_full, baseline

    def _refine_gates_with_tracking(
        self,
        video_path,
        initial_gates,
        conf=0.35,
        iou=0.55,
        max_frames=120,
        stride=3,
        min_obs=3,
    ):
        """
        Run temporal gate tracking over early frames to stabilize detections.
        Returns median gate positions for each tracked gate.
        """
        if not initial_gates:
            return initial_gates

        tracker = TemporalGateTracker()
        tracker.initialize(initial_gates)

        history = {}
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        stride = max(1, int(stride))
        max_frames = max(1, int(max_frames))

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            detections = self.gate_detector.detect_in_frame(frame, conf=conf, iou=iou)
            tracked = tracker.update(detections, frame_idx=frame_idx)
            for g in tracked:
                history.setdefault(g["gate_id"], []).append(
                    (g["center_x"], g["base_y"], g["class"],
                     g.get("class_name", "gate"), g.get("confidence", 0.0))
                )

            frame_idx += 1

        cap.release()

        refined = []
        for gate_id, items in history.items():
            if len(items) < min_obs:
                continue
            xs = [i[0] for i in items]
            ys = [i[1] for i in items]
            cls = items[0][2]
            name = items[0][3]
            confs = [i[4] for i in items if len(i) > 4]
            refined.append({
                "gate_id": gate_id,
                "center_x": float(np.median(xs)),
                "base_y": float(np.median(ys)),
                "class": int(cls),
                "class_name": name,
                "confidence": float(np.median(confs)) if confs else 0.0,
            })

        if not refined:
            return initial_gates

        refined.sort(key=lambda g: g["base_y"])
        return refined

    def _auto_calibrate_scale(self, trajectory_3d, fps, trajectory_2d):
        """
        Physics-based scale auto-calibration.

        If the high-percentile speed from the initial 3D transform exceeds what's
        physically possible for the discipline, compute a correction factor
        and rescale all 3D coordinates.

        Expected median speeds by discipline:
          - Slalom:       30-45 km/h (max ~70 km/h)
          - Giant Slalom: 40-60 km/h (max ~100 km/h)

        The correction factor = current_p90_speed / target_p90_speed.
        We divide all 3D coordinates by this factor to bring speeds into
        the physically realistic range.
        """
        if len(trajectory_3d) < 10:
            return trajectory_3d, {
                "applied": False,
                "unable_to_calibrate": True,
                "reason": "too_few_points",
                "correction": None,
                "correction_factor": None,
            }

        # Compute median speed from current 3D trajectory
        speeds = []
        for i in range(1, len(trajectory_3d)):
            dx = trajectory_3d[i]["x"] - trajectory_3d[i - 1]["x"]
            dy = trajectory_3d[i]["y"] - trajectory_3d[i - 1]["y"]
            dist = (dx**2 + dy**2) ** 0.5
            dt = (trajectory_3d[i]["frame"] - trajectory_3d[i - 1]["frame"]) / fps
            if dt > 0:
                speeds.append(dist / dt * 3.6)  # km/h

        if not speeds:
            return trajectory_3d, {
                "applied": False,
                "unable_to_calibrate": True,
                "reason": "no_speeds",
                "correction": None,
                "correction_factor": None,
            }

        speeds_np = np.asarray(speeds, dtype=float)
        # Drop exact zeros (stationary/undetected) for calibration stats
        speeds_nz = speeds_np[speeds_np > 1e-3]
        if len(speeds_nz) < max(10, int(0.5 * len(speeds_np))):
            speeds_nz = speeds_np  # fallback if too many zeros

        p90_speed = float(np.percentile(speeds_nz, 90))
        p95_speed = float(np.percentile(speeds_nz, 95))
        median_speed = float(np.median(speeds_nz))

        # Target high-percentile speeds by discipline (used for calibration).
        # We use P90 (not median) to avoid zeros/undetected frames masking scale explosions.
        DISCIPLINE_TARGETS = {
            "slalom":       {"target_p90": 55.0, "max_reasonable_p90": 70.0, "target_median": 38.0},
            "giant_slalom": {"target_p90": 80.0, "max_reasonable_p90": 100.0, "target_median": 50.0},
        }

        target = DISCIPLINE_TARGETS.get(self.discipline, DISCIPLINE_TARGETS["slalom"])
        max_p90 = target["max_reasonable_p90"]

        if p90_speed <= max_p90:
            # Scale is already reasonable
            return trajectory_3d, {
                "applied": False,
                "unable_to_calibrate": False,
                "reason": "p90_within_limit",
                "correction": 1.0,
                "correction_factor": 1.0,
                "raw_correction_factor": 1.0,
                "correction_cap": 5.0,
                "median_speed": median_speed,
                "p90_speed": p90_speed,
                "p95_speed": p95_speed,
            }

        # Compute correction factor
        target_p90 = target["target_p90"]
        raw_correction = float(max(1e-6, p90_speed / target_p90))
        correction_cap = 5.0
        likely_causes = [
            "wrong gate spacing",
            "wrong camera pitch",
            "too few gates detected",
            "gate detections on non-gate objects",
        ]

        print(f"  ⚠️  AUTO-CALIBRATION: P90 speed {p90_speed:.0f} km/h exceeds "
              f"{max_p90:.0f} km/h limit for {self.discipline}")

        if raw_correction > 3.0:
            print(f"     Large correction requested: {raw_correction:.2f}× (>3×). "
                  f"Likely causes: {', '.join(likely_causes)}")

        if raw_correction > correction_cap:
            print(f"  ❌ Unable to calibrate: requested correction {raw_correction:.2f}× exceeds "
                  f"{correction_cap:.1f}× cap. Upstream geometry is likely broken.")
            print(f"     Diagnostics: likely causes -> {', '.join(likely_causes)}")
            return trajectory_3d, {
                "applied": False,
                "unable_to_calibrate": True,
                "reason": "correction_exceeds_cap",
                "correction": raw_correction,
                "correction_factor": raw_correction,
                "raw_correction_factor": raw_correction,
                "correction_cap": correction_cap,
                "median_speed": median_speed,
                "p90_speed": p90_speed,
                "p95_speed": p95_speed,
                "target_p90": target_p90,
                "max_reasonable_p90": max_p90,
                "n_speeds": int(len(speeds_np)),
                "n_nonzero": int(len(speeds_nz)),
                "likely_causes": likely_causes,
            }

        correction = raw_correction

        print(f"     Applying {correction:.2f}× scale correction "
              f"(target P90: {target_p90:.0f} km/h)")
        print(f"     This suggests the gate-based scale was off by ~{correction:.1f}×")

        # Rescale: divide all 3D coordinates by the correction factor.
        # This is equivalent to multiplying ppm by the correction factor.
        # We rescale relative to the first point to keep the origin stable.
        x0 = trajectory_3d[0]["x"]
        y0 = trajectory_3d[0]["y"]

        for pt in trajectory_3d:
            pt["x"] = x0 + (pt["x"] - x0) / correction
            pt["y"] = y0 + (pt["y"] - y0) / correction

        # Also update the transformer's ppm_y for consistency
        if self.transformer.ppm_y is not None:
            old_ppm = self.transformer.ppm_y
            self.transformer.ppm_y *= correction
            print(f"     ppm_y corrected: {old_ppm:.2f} → {self.transformer.ppm_y:.2f} px/m")

        return trajectory_3d, {
            "applied": True,
            "unable_to_calibrate": False,
            "correction": correction,
            "correction_factor": correction,
            "raw_correction_factor": raw_correction,
            "correction_cap": correction_cap,
            "median_speed": median_speed,
            "p90_speed": p90_speed,
            "p95_speed": p95_speed,
            "target_p90": target_p90,
            "max_reasonable_p90": max_p90,
            "n_speeds": int(len(speeds_np)),
            "n_nonzero": int(len(speeds_nz)),
        }

    @staticmethod
    def _sanitize_trajectory_3d(trajectory_3d, fps,
                                hard_jump_m=10.0,
                                soft_jump_m=5.0):
        """
        Remove pathological points before calibration/physics.

        NOTE: Jump-based clamping has been removed in favor of Mahalanobis
        innovation gating in the Kalman smoother. This sanitizer now only
        removes non-finite points and interpolates them.

        Returns:
            (trajectory_3d, info_dict)
        """
        if len(trajectory_3d) < 3:
            return trajectory_3d, {
                "hard_removed": 0,
                "soft_clamped": 0,
                "hard_jump_m": hard_jump_m,
                "soft_jump_m": soft_jump_m,
            }

        traj = [dict(p) for p in trajectory_3d]
        removed = 0
        non_finite = 0
        interpolated = 0

        # Mark non-finite points as invalid
        valid = [True] * len(traj)
        for i in range(len(traj)):
            curr = traj[i]
            if not np.isfinite(curr["x"]) or not np.isfinite(curr["y"]):
                valid[i] = False
                non_finite += 1
                removed += 1

        # Interpolate invalid points between nearest valid anchors
        last_valid = None
        i = 0
        while i < len(traj):
            if valid[i]:
                last_valid = i
                i += 1
                continue
            # find next valid
            j = i + 1
            while j < len(traj) and not valid[j]:
                j += 1
            if last_valid is None or j >= len(traj):
                # Can't interpolate at edges; skip
                i = j
                continue
            # interpolate between last_valid and j
            f0 = traj[last_valid]["frame"]
            f1 = traj[j]["frame"]
            x0 = traj[last_valid]["x"]
            y0 = traj[last_valid]["y"]
            x1 = traj[j]["x"]
            y1 = traj[j]["y"]
            for k in range(i, j):
                fk = traj[k]["frame"]
                t = (fk - f0) / (f1 - f0) if f1 > f0 else 0.5
                traj[k]["x"] = x0 + t * (x1 - x0)
                traj[k]["y"] = y0 + t * (y1 - y0)
                valid[k] = True
                interpolated += 1
            i = j

        return traj, {
            "hard_removed": 0,
            "soft_clamped": 0,
            "non_finite_removed": non_finite,
            "interpolated": interpolated,
            "hard_jump_m": None,
            "soft_jump_m": None,
        }

    @staticmethod
    def _smooth_trajectory_3d(trajectory_3d, window=5):
        """
        Median smooth 3D trajectory to reduce jitter before physics.

        Uses a small window median filter on x and y. This is robust to
        outliers and preserves sharp turns better than a mean filter.
        """
        if len(trajectory_3d) < 3:
            return trajectory_3d

        window = int(window)
        if window < 3:
            return trajectory_3d
        if window % 2 == 0:
            window += 1

        half = window // 2
        smoothed = []
        n = len(trajectory_3d)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n - 1, i + half)
            xs = [trajectory_3d[j]["x"] for j in range(lo, hi + 1)]
            ys = [trajectory_3d[j]["y"] for j in range(lo, hi + 1)]
            smoothed.append({
                "frame": trajectory_3d[i]["frame"],
                "x": float(np.median(xs)),
                "y": float(np.median(ys)),
            })

        return smoothed

    def _cluster_gates_by_y(self, gates, y_thresh=12.0, frame_height=None):
        """
        Cluster detections with similar base_y to avoid duplicate gate poles.

        Two poles of the same slalom gate can be 20-40px apart vertically in
        typical footage. The adaptive threshold uses 8% of frame height to
        account for this, plus a gap-based threshold (45% of median gap).
        """
        if not gates:
            return gates

        gates_sorted = sorted(gates, key=lambda g: g["base_y"])

        # Adaptive threshold: 8% of frame height (two poles of same gate
        # can be ~20-40px apart vertically in typical footage)
        if frame_height and frame_height > 0:
            adaptive_thresh = 0.08 * frame_height
        else:
            adaptive_thresh = y_thresh

        # Dynamic threshold from gap statistics
        gaps = []
        for i in range(1, len(gates_sorted)):
            dy = gates_sorted[i]["base_y"] - gates_sorted[i - 1]["base_y"]
            if dy > 1e-3:
                gaps.append(dy)
        if gaps:
            median_gap = float(np.median(gaps))
            gap_thresh = 0.45 * median_gap  # was 0.35, too conservative
        else:
            gap_thresh = y_thresh

        # Use the LARGER of the two thresholds
        dynamic_thresh = max(adaptive_thresh, gap_thresh)
        clusters = []
        current = [gates_sorted[0]]

        for g in gates_sorted[1:]:
            avg_y = sum(x["base_y"] for x in current) / len(current)
            if abs(g["base_y"] - avg_y) <= dynamic_thresh:
                current.append(g)
            else:
                clusters.append(current)
                current = [g]

        if current:
            clusters.append(current)

        merged = []
        for group in clusters:
            xs = [g["center_x"] for g in group]
            ys = [g["base_y"] for g in group]
            cls_list = [g.get("class", 0) for g in group]
            cls = max(set(cls_list), key=cls_list.count)
            name = next((g.get("class_name", "gate") for g in group if g.get("class", 0) == cls), "gate")
            confs = [g.get("confidence", 0.0) for g in group if g.get("confidence") is not None]
            merged.append({
                "center_x": float(np.median(xs)),
                "base_y": float(np.median(ys)),
                "class": int(cls),
                "class_name": name,
                "confidence": float(np.median(confs)) if confs else 0.0,
            })

        return merged


def main():
    """CLI entry point for processing videos."""
    import argparse

    parser = argparse.ArgumentParser(description="Process ski racing video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--gate-model", required=True, help="Path to gate detector model")
    parser.add_argument("--discipline", default=None,
                        choices=["slalom", "giant_slalom"],
                        help="Discipline. If omitted, auto-detects slalom vs giant_slalom from gates")
    parser.add_argument("--gate-spacing", type=float, default=None,
                        help=("Gate spacing in meters. If omitted, uses discipline defaults: "
                              "slalom=9.5, giant_slalom=27"))
    parser.add_argument("--output-dir", default="artifacts/outputs", help="Output directory")
    parser.add_argument("--gate-conf", type=float, default=0.35,
                        help="Gate detection confidence threshold")
    parser.add_argument("--gate-iou", type=float, default=0.55,
                        help="Gate detection NMS IoU threshold")
    parser.add_argument("--skier-conf", type=float, default=0.25,
                        help="Skier detection confidence threshold")
    parser.add_argument("--no-physics", action="store_true",
                        help="Skip physics validation")
    parser.add_argument("--stabilize", action="store_true",
                        help="Enable camera stabilization + smoothing + dynamic scale")
    parser.add_argument("--camera-mode", default="affine",
                        choices=["translation", "affine"],
                        help="Camera motion model for stabilization (affine or translation)")
    parser.add_argument("--camera-pitch-deg", "--camera-pitch", dest="camera_pitch_deg",
                        type=float, default=6.0,
                        help="Camera pitch angle in degrees for scale correction (set 0 to disable)")
    parser.add_argument("--kalman-q", type=float, default=None,
                        help="Override Kalman process noise (Q sigma_a in px/s^2)")
    parser.add_argument("--camera-compensate-before-smoothing", action="store_true",
                        help="Experimental order: track -> camera compensate 2D -> smooth -> transform")
    parser.add_argument("--disable-dynamic-scale", action="store_true",
                        help="Disable Phase 4 dynamic per-frame scale")
    args = parser.parse_args()

    pipeline = SkiRacingPipeline(
        gate_model_path=args.gate_model,
        discipline=args.discipline,
        gate_spacing_m=args.gate_spacing,
        stabilize=args.stabilize,
        camera_mode=args.camera_mode,
        camera_pitch_deg=args.camera_pitch_deg,
    )

    results = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        validate_physics=not args.no_physics,
        gate_conf=args.gate_conf,
        gate_iou=args.gate_iou,
        skier_conf=args.skier_conf,
        kalman_process_noise=args.kalman_q,
        camera_compensate_before_smoothing=args.camera_compensate_before_smoothing,
        enable_dynamic_scale=not args.disable_dynamic_scale,
    )


if __name__ == "__main__":
    main()
