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
from pathlib import Path
from datetime import datetime

from .detection import GateDetector, TemporalGateTracker
from .tracking import SkierTracker, KalmanSmoother
from .transform import HomographyTransform, CameraMotionCompensator, DynamicScaleTransform
from .physics import PhysicsValidator


class SkiRacingPipeline:
    """
    End-to-end pipeline: video -> gates + 3D trajectory + physics validation.
    """

    def __init__(self, gate_model_path, discipline="slalom", gate_spacing_m=12.0,
                 stabilize=False, camera_mode="translation"):
        """
        Args:
            gate_model_path: Path to trained YOLOv8 gate detection weights.
            discipline: "slalom", "giant_slalom", or "downhill".
            gate_spacing_m: Expected distance between gates in meters.
            stabilize: Enable all 4 improvement phases (camera compensation,
                      Kalman smoothing, dynamic scale, per-frame gates).
            camera_mode: "translation" (default) or "affine" for camera motion.
        """
        self.gate_detector = GateDetector(gate_model_path)
        self.skier_tracker = SkierTracker()
        self.transformer = HomographyTransform()
        self.discipline = discipline
        self.gate_spacing_m = gate_spacing_m
        self.stabilize = stabilize
        self.camera_mode = camera_mode

    def process_video(
        self,
        video_path,
        output_dir="artifacts/outputs",
        validate_physics=True,
        projection="scale",
        gate_conf=0.3,
        gate_search_frames=150,
        gate_search_stride=5,
        gate_track_frames=120,
        gate_track_stride=3,
        gate_track_min_obs=3,
        frame_stride=1,
        max_frames=None,
        max_jump=None,
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

        # ─── Step 1: Detect gates from first/best frame ───
        print(f"  [1/{n_steps}] Detecting gates...")
        gates = self.gate_detector.detect_from_first_frame(video_path, conf=gate_conf)
        if len(gates) < 4 and gate_search_frames > 0:
            print("         Low gate count; scanning additional frames...")
            gates = self.gate_detector.detect_from_best_frame(
                video_path,
                conf=gate_conf,
                max_frames=gate_search_frames,
                stride=gate_search_stride,
            )

        # ─── Step 1b: Per-frame gate tracking (Phase 1) ───
        frame_gate_history = None
        baseline_gates = None

        if self.stabilize and len(gates) >= 2:
            print(f"  [2/{n_steps}] Tracking gates across all frames (Phase 1)...")
            gates, frame_gate_history, baseline_gates = self._track_gates_full_video(
                video_path,
                gates,
                conf=gate_conf,
                stride=gate_track_stride,
            )
        elif gate_track_frames > 0 and len(gates) >= 2:
            print("         Stabilizing gates with temporal tracking...")
            gates = self._refine_gates_with_tracking(
                video_path,
                gates,
                conf=gate_conf,
                max_frames=gate_track_frames,
                stride=gate_track_stride,
                min_obs=gate_track_min_obs,
            )

        if len(gates) >= 2:
            gates = self._cluster_gates_by_y(gates, y_thresh=12.0)
        print(f"         Found {len(gates)} gates")

        # ─── Step 2/3: Calculate projection transform ───
        step_n = 3 if self.stabilize else 2
        print(f"  [{step_n}/{n_steps}] Calculating perspective transform...")
        gate_centers = [[g["center_x"], g["base_y"]] for g in gates]
        if projection == "scale":
            self.transformer.calculate_scale_from_gates(gate_centers, self.gate_spacing_m)
        else:
            self.transformer.calculate_from_gates(gate_centers, self.gate_spacing_m)

        # ─── Phase 2: Camera motion compensation ───
        if self.stabilize and frame_gate_history and baseline_gates:
            print(f"  [{step_n}/{n_steps}]   + Camera motion compensation (Phase 2)...")
            compensator = CameraMotionCompensator(
                baseline_gates,
                frame_gate_history,
                mode=self.camera_mode,
            )
            compensator.estimate_motion()
            self.transformer.camera_compensator = compensator

        # ─── Phase 4: Dynamic scale ───
        if self.stabilize and frame_gate_history:
            print(f"  [{step_n}/{n_steps}]   + Dynamic per-frame scale (Phase 4)...")
            dyn_scale = DynamicScaleTransform(frame_gate_history, self.gate_spacing_m)
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

        # ─── Step 3/4: Track skier ───
        step_n = 4 if self.stabilize else 3
        print(f"  [{step_n}/{n_steps}] Tracking skier...")
        trajectory_2d = self.skier_tracker.track_video(
            video_path,
            method="bytetrack",
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_jump=max_jump,
        )
        print(f"         Tracked {len(trajectory_2d)} positions "
              f"({len(trajectory_2d)/total_frames*100:.0f}% coverage)")

        # Keep raw trajectory for comparison
        trajectory_2d_raw = None

        # ─── Phase 3: Kalman smoothing ───
        if self.stabilize and len(trajectory_2d) > 2:
            step_n = 5
            print(f"  [{step_n}/{n_steps}] Smoothing trajectory with Kalman filter (Phase 3)...")
            trajectory_2d_raw = [dict(pt) for pt in trajectory_2d]  # Deep copy
            kf = KalmanSmoother(
                fps=fps,
                discipline=self.discipline,
            )
            trajectory_2d = kf.smooth(trajectory_2d)

        # ─── Step 4/6: Transform to 3D ───
        step_n = 6 if self.stabilize else 4
        print(f"  [{step_n}/{n_steps}] Transforming to 3D coordinates...")
        trajectory_3d = self.transformer.transform_trajectory(
            trajectory_2d,
            stabilize=self.stabilize,
            fps=fps,
        )

        # ─── Step 5/7: Physics validation ───
        step_n = 7 if self.stabilize else 5
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

        # Compile results
        results = {
            "video": video_path,
            "video_info": video_info,
            "discipline": self.discipline,
            "gate_spacing_m": self.gate_spacing_m,
            "projection": projection,
            "stabilized": self.stabilize,
            "camera_mode": self.camera_mode if self.stabilize else None,
            "gates": gates,
            "trajectory_2d": trajectory_2d,
            "trajectory_3d": trajectory_3d,
            "physics_validation": physics_result,
            "timestamp": datetime.now().isoformat(),
        }

        # Add stabilization-specific data
        if self.stabilize:
            results["kalman_smoothed"] = True
            if trajectory_2d_raw is not None:
                results["trajectory_2d_raw"] = trajectory_2d_raw
            if self.transformer.camera_compensator:
                if self.camera_mode == "affine":
                    results["camera_motion_frames"] = len(self.transformer.camera_compensator.affine)
                else:
                    results["camera_motion_frames"] = len(self.transformer.camera_compensator.offsets)
            if frame_gate_history:
                results["frames"] = self._build_frame_records(frame_gate_history)
                results["gate_track_stride"] = gate_track_stride

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
            for gate_id, (cx, by) in frame_gate_history[frame_idx].items():
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

    def _track_gates_full_video(self, video_path, initial_gates, conf=0.3, stride=3):
        """
        Track gates across the entire video to build per-frame gate history.

        Returns:
            Tuple of (refined_gates, frame_gate_history, baseline_gates)
            - refined_gates: List of gate dicts with median positions
            - frame_gate_history: {frame_idx: {gate_id: (cx, by)}}
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

            detections = self.gate_detector.detect_in_frame(frame, conf=conf)
            tracked = tracker.update_with_frame_idx(detections, frame_idx)

            for g in tracked:
                if not g.get("is_interpolated", False):
                    history_for_median.setdefault(g["gate_id"], []).append(
                        (g["center_x"], g["base_y"], g["class"], g.get("class_name", "gate"))
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
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            refined.append({
                "gate_id": gate_id,
                "center_x": med_x,
                "base_y": med_y,
                "class": int(cls),
                "class_name": name,
                "confidence": 1.0,
            })
            baseline[gate_id] = (med_x, med_y)

        if not refined:
            return initial_gates, {}, {}

        refined.sort(key=lambda g: g["base_y"])
        frame_gate_history = tracker.get_frame_history()

        print(f"         Tracked {len(refined)} gates across {processed} frames "
              f"({len(frame_gate_history)} frames with detections)")

        return refined, frame_gate_history, baseline

    def _refine_gates_with_tracking(
        self,
        video_path,
        initial_gates,
        conf=0.3,
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

            detections = self.gate_detector.detect_in_frame(frame, conf=conf)
            tracked = tracker.update(detections)
            for g in tracked:
                history.setdefault(g["gate_id"], []).append(
                    (g["center_x"], g["base_y"], g["class"], g.get("class_name", "gate"))
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
            refined.append({
                "gate_id": gate_id,
                "center_x": float(np.median(xs)),
                "base_y": float(np.median(ys)),
                "class": int(cls),
                "class_name": name,
                "confidence": 1.0,
            })

        if not refined:
            return initial_gates

        refined.sort(key=lambda g: g["base_y"])
        return refined

    def _cluster_gates_by_y(self, gates, y_thresh=12.0):
        """
        Cluster detections with similar base_y to avoid duplicate gate poles.
        """
        if not gates:
            return gates

        gates_sorted = sorted(gates, key=lambda g: g["base_y"])
        clusters = []
        current = [gates_sorted[0]]

        for g in gates_sorted[1:]:
            avg_y = sum(x["base_y"] for x in current) / len(current)
            if abs(g["base_y"] - avg_y) <= y_thresh:
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
            cls = group[0].get("class", 0)
            name = group[0].get("class_name", "gate")
            merged.append({
                "center_x": float(np.median(xs)),
                "base_y": float(np.median(ys)),
                "class": int(cls),
                "class_name": name,
                "confidence": 1.0,
            })

        return merged


def main():
    """CLI entry point for processing videos."""
    import argparse

    parser = argparse.ArgumentParser(description="Process ski racing video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--gate-model", required=True, help="Path to gate detector model")
    parser.add_argument("--discipline", default="slalom",
                        choices=["slalom", "giant_slalom", "downhill"])
    parser.add_argument("--gate-spacing", type=float, default=12.0,
                        help="Gate spacing in meters")
    parser.add_argument("--output-dir", default="artifacts/outputs", help="Output directory")
    parser.add_argument("--no-physics", action="store_true",
                        help="Skip physics validation")
    parser.add_argument("--stabilize", action="store_true",
                        help="Enable camera stabilization + smoothing + dynamic scale")
    parser.add_argument("--camera-mode", default="translation",
                        choices=["translation", "affine"],
                        help="Camera motion model for stabilization (translation or affine)")
    args = parser.parse_args()

    pipeline = SkiRacingPipeline(
        gate_model_path=args.gate_model,
        discipline=args.discipline,
        gate_spacing_m=args.gate_spacing,
        stabilize=args.stabilize,
        camera_mode=args.camera_mode,
    )

    results = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        validate_physics=not args.no_physics,
    )


if __name__ == "__main__":
    main()
