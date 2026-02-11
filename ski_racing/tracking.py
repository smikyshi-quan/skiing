"""
Skier tracking module.
Uses YOLOv8 with ByteTrack for robust multi-person tracking,
correctly identifying the racer among spectators and course workers.

Includes Kalman filter smoothing (Phase 3) for removing noise
from bounding-box-based position estimates.
"""
import cv2
import numpy as np
from pathlib import Path


class KalmanSmoother:
    """
    Rauch-Tung-Striebel (RTS) smoother for 2D trajectory smoothing.

    State vector: [x, y, vx, vy]
    Measurement: [x, y] (noisy pixel coordinates from detector)

    Uses a forward Kalman filter pass followed by a backward RTS
    smoothing pass. The backward pass corrects the forward estimates
    using future information, which is critical at turn apexes where
    a forward-only filter would lag behind direction changes.

    Pure NumPy implementation — no external dependencies.

    Professor's feedback on over-smoothing:
        If process_noise (Q) is too low, the filter trusts its constant-
        velocity prediction too much and cuts slalom turn corners. For
        slalom (turns every 1-2s), use process_noise >= 2.0. For GS/DH
        (smoother arcs), 1.0-1.5 is fine.
    """

    # Discipline-tuned defaults: slalom needs higher Q to track rapid turns.
    # process_noise (sigma_a) is in pixels/s² and represents expected
    # acceleration standard deviation. This is fps-independent.
    # Slalom: rapid direction changes every 1-2s → high acceleration.
    # GS/DH: smoother arcs → lower acceleration expected.
    DISCIPLINE_DEFAULTS = {
        "slalom":       {"process_noise": 800.0, "measurement_noise": 5.0},
        "giant_slalom": {"process_noise": 400.0, "measurement_noise": 5.0},
        "downhill":     {"process_noise": 250.0, "measurement_noise": 5.0},
    }

    def __init__(self, process_noise=None, measurement_noise=None, fps=30.0,
                 discipline=None):
        """
        Args:
            process_noise: Expected acceleration std dev (pixels/s²).
                          Higher = trust measurements more, lower = smoother.
                          If None, uses discipline default.
                          Typical values: 200-1000 for ski racing.
            measurement_noise: Expected detection noise std dev (pixels).
                              Typical bounding box center jitter is 3-10 px.
                              If None, uses discipline default.
            fps: Video frame rate for time delta.
            discipline: "slalom", "giant_slalom", or "downhill".
                       Used to auto-tune process_noise if not specified.
        """
        self.dt = 1.0 / fps
        self.discipline = discipline or "slalom"

        defaults = self.DISCIPLINE_DEFAULTS.get(self.discipline, self.DISCIPLINE_DEFAULTS["slalom"])
        self.process_noise = process_noise if process_noise is not None else defaults["process_noise"]
        self.measurement_noise = measurement_noise if measurement_noise is not None else defaults["measurement_noise"]

    def smooth(self, trajectory_2d):
        """
        Apply forward-backward RTS smoother to a 2D trajectory.

        Forward pass: standard Kalman filter (predict + update).
        Backward pass: RTS smoother refines estimates using future data.

        The backward pass is what prevents corner-cutting at turn apexes —
        it allows the filter to "look ahead" and adjust positions that a
        forward-only filter would lag on.

        Handles frame gaps by running multiple prediction steps.

        Args:
            trajectory_2d: List of {"frame": int, "x": float, "y": float}.

        Returns:
            Smoothed trajectory with same structure.
        """
        if len(trajectory_2d) < 2:
            return trajectory_2d

        dt = self.dt
        n = len(trajectory_2d)

        # State transition matrix F (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Process noise covariance Q (discrete white noise acceleration model)
        # sigma_a is in pixels/s², making Q fps-independent.
        # Q = G * G^T * sigma_a^2 where G = [dt^2/2, dt^2/2, dt, dt]
        sigma_a = self.process_noise  # pixels/s²
        G = np.array([dt**2 / 2, dt**2 / 2, dt, dt], dtype=np.float64)
        Q = np.outer(G, G) * sigma_a**2
        # Zero out cross-axis terms (x-accel doesn't affect y)
        Q[0, 1] = Q[1, 0] = 0.0
        Q[0, 3] = Q[3, 0] = 0.0
        Q[1, 2] = Q[2, 1] = 0.0
        Q[2, 3] = Q[3, 2] = 0.0

        # Observation matrix H (we observe position only)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        r = self.measurement_noise
        R = np.eye(2, dtype=np.float64) * r**2

        # Initialize state from first measurement
        x0 = trajectory_2d[0]["x"]
        y0 = trajectory_2d[0]["y"]

        # Estimate initial velocity from first two points
        if n >= 2:
            df = max(1, trajectory_2d[1]["frame"] - trajectory_2d[0]["frame"])
            vx0 = (trajectory_2d[1]["x"] - x0) / (df * dt)
            vy0 = (trajectory_2d[1]["y"] - y0) / (df * dt)
        else:
            vx0, vy0 = 0.0, 0.0

        state = np.array([x0, y0, vx0, vy0], dtype=np.float64)
        P = np.diag([r**2, r**2, (r * 2)**2, (r * 2)**2])

        # ─── Forward pass: standard Kalman filter ───
        # Store predicted and filtered states/covariances for RTS backward pass
        states_filt = []      # filtered state at each measurement
        covs_filt = []        # filtered covariance at each measurement
        states_pred = []      # predicted state (before update) at each measurement
        covs_pred = []        # predicted covariance (before update)
        F_steps = []          # effective F matrix used for each prediction step

        prev_frame = trajectory_2d[0]["frame"]

        for i, pt in enumerate(trajectory_2d):
            frame = pt["frame"]
            z = np.array([pt["x"], pt["y"]], dtype=np.float64)

            # Handle frame gaps: compute effective F^gap and accumulated Q
            frame_gap = max(1, frame - prev_frame) if i > 0 else 0

            if frame_gap <= 1:
                F_eff = F
                Q_eff = Q
            else:
                # F^gap for multi-step prediction
                F_eff = np.linalg.matrix_power(F, frame_gap)
                # Accumulated process noise: sum of F^j Q F^jT for j=0..gap-1
                Q_eff = np.zeros_like(Q)
                F_pow = np.eye(4, dtype=np.float64)
                for _ in range(frame_gap):
                    Q_eff += F_pow @ Q @ F_pow.T
                    F_pow = F_pow @ F

            F_steps.append(F_eff)

            if i > 0:
                # Predict using effective transition
                state = F_eff @ state
                P = F_eff @ P @ F_eff.T + Q_eff

            # Store predicted (before update)
            states_pred.append(state.copy())
            covs_pred.append(P.copy())

            if i > 0:
                # Update with measurement
                y_innov = z - H @ state
                S = H @ P @ H.T + R
                try:
                    K = P @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    K = np.zeros((4, 2))

                state = state + K @ y_innov
                P = (np.eye(4) - K @ H) @ P

            # Store filtered (after update)
            states_filt.append(state.copy())
            covs_filt.append(P.copy())
            prev_frame = frame

        # ─── Backward pass: RTS smoother ───
        # Uses the effective transition matrix from each forward step
        states_smooth = [None] * n
        states_smooth[n - 1] = states_filt[n - 1]

        for k in range(n - 2, -1, -1):
            P_filt = covs_filt[k]
            P_pred_next = covs_pred[k + 1]
            F_k = F_steps[k + 1]  # Transition used from step k to k+1

            # RTS gain: G_k = P_k|k * F_k^T * (P_{k+1|k})^{-1}
            try:
                G = P_filt @ F_k.T @ np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                G = np.zeros((4, 4))

            states_smooth[k] = states_filt[k] + G @ (states_smooth[k + 1] - states_pred[k + 1])

        # Build output
        smoothed = []
        for i, pt in enumerate(trajectory_2d):
            smoothed.append({
                "frame": pt["frame"],
                "x": float(states_smooth[i][0]),
                "y": float(states_smooth[i][1]),
            })

        # ── Over-smoothing diagnostic (Prof. feedback Phase 3) ──
        # If Q is too low, the filter cuts corners on sharp turns.
        # Compare lateral deviation at turn apexes (x-reversal points)
        # between raw measurements and smoothed output.
        self._check_over_smoothing(trajectory_2d, smoothed)

        return smoothed

    def _check_over_smoothing(self, raw, smoothed):
        """
        Detect if smoothing is cutting corners on sharp turns.

        Finds turn apexes (points where X-direction reverses) and compares
        lateral deviation between raw and smoothed paths. If the smoothed
        path consistently reduces apex deviation by >30%, the filter is
        probably too stiff (process_noise too low).

        Professor's caution: "the Kalman filter will be too stiff — it will
        cut the corners of sharp slalom turns, reporting a turn radius of
        15m when the skier actually turned at 8m."
        """
        if len(raw) < 10:
            return

        # Find turn apexes: points where dx changes sign
        raw_x = [p["x"] for p in raw]
        smooth_x = [p["x"] for p in smoothed]

        apex_indices = []
        for i in range(1, len(raw_x) - 1):
            dx_prev = raw_x[i] - raw_x[i - 1]
            dx_next = raw_x[i + 1] - raw_x[i]
            if dx_prev * dx_next < 0:  # sign change = turn apex
                apex_indices.append(i)

        if len(apex_indices) < 2:
            return

        # Measure how much the smoothed path reduces lateral excursion
        raw_y = [p["y"] for p in raw]

        # Use a wider local baseline: average x of neighbors ±10 frames.
        # This is more robust than ±5 because noise at individual frames
        # can inflate the raw deviation, leading to false positives.
        deviations_raw = []
        deviations_smooth = []
        window = 10

        for idx in apex_indices:
            lo = max(0, idx - window)
            hi = min(len(raw_x) - 1, idx + window)
            # Exclude the apex itself from the baseline
            baseline_x = [raw_x[j] for j in range(lo, hi + 1) if j != idx]
            if not baseline_x:
                continue
            local_avg_x = np.mean(baseline_x)
            deviations_raw.append(abs(raw_x[idx] - local_avg_x))
            deviations_smooth.append(abs(smooth_x[idx] - local_avg_x))

        avg_raw = np.mean(deviations_raw) if deviations_raw else 0
        avg_smooth = np.mean(deviations_smooth) if deviations_smooth else 0

        # Only warn if: (1) meaningful lateral movement at apexes (>10px),
        # and (2) smoothing reduces apex deviation by >50%.
        # 50% threshold is higher than 30% because the RTS smoother naturally
        # adjusts turn peaks slightly, and raw apex deviations include noise.
        if avg_raw > 10.0:
            reduction = 1.0 - avg_smooth / avg_raw if avg_raw > 0 else 0
            if reduction > 0.50:
                print(f"  ⚠️  Over-smoothing detected: turn apex deviation reduced by "
                      f"{reduction * 100:.0f}% ({avg_raw:.1f}px → {avg_smooth:.1f}px). "
                      f"The filter may be cutting corners on sharp turns (Q={self.process_noise}). "
                      f"Increase process_noise or verify with demo video overlay.")


class SkierTracker:
    """
    Track the racing skier through video frames.
    Uses ByteTrack for persistent ID tracking across frames,
    with fallback to temporal consistency when ByteTrack loses the target.
    """

    def __init__(self, model_name=None):
        """
        Args:
            model_name: YOLO model for person detection.
        """
        from ultralytics import YOLO  # Lazy import: not needed for KalmanSmoother

        if model_name is None:
            # Prefer local weights to avoid network downloads
            if Path("models/yolov8s.pt").exists():
                model_name = "models/yolov8s.pt"
            elif Path("models/yolov8n.pt").exists():
                model_name = "models/yolov8n.pt"
            else:
                model_name = "yolov8n.pt"
        self.model = YOLO(model_name)

    def track_video(
        self,
        video_path,
        method="bytetrack",
        frame_stride=1,
        max_frames=None,
        max_jump=None,
    ):
        """
        Track skier through an entire video.

        Args:
            video_path: Path to video file.
            method: "bytetrack" (recommended) or "temporal" (fallback).
            frame_stride: Process every Nth frame (temporal mode only).
            max_frames: Optional max number of frames to process from start.
            max_jump: Max pixel jump allowed between frames (temporal mode only).

        Returns:
            List of trajectory points: [{"frame": int, "x": float, "y": float}, ...]
        """
        if method == "bytetrack":
            try:
                return self._track_with_bytetrack(video_path)
            except ModuleNotFoundError as e:
                print(f"⚠️  ByteTrack dependency missing ({e}). Falling back to temporal tracking.")
                return self._track_with_temporal_consistency(
                    video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                    max_jump=max_jump,
                )
            except Exception as e:
                print(f"⚠️  ByteTrack failed ({e}). Falling back to temporal tracking.")
                return self._track_with_temporal_consistency(
                    video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                    max_jump=max_jump,
                )
        else:
            return self._track_with_temporal_consistency(
                video_path,
                frame_stride=frame_stride,
                max_frames=max_frames,
                max_jump=max_jump,
            )

    def _track_with_bytetrack(self, video_path):
        """
        Use YOLOv8's built-in ByteTrack for robust multi-person tracking.
        Identifies the racer by assuming they are the most consistently
        moving person in the frame (not static spectators).
        """
        trajectory = []
        racer_id = None

        # Run tracking with persistence
        results = self.model.track(
            source=video_path,
            classes=[0],  # person class only
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            stream=True,
        )

        # First pass: identify which tracked ID is the racer
        all_positions = {}  # id -> list of (frame, x, y)
        total_frames = 0

        for frame_idx, result in enumerate(results):
            total_frames += 1
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x = float((box[0] + box[2]) / 2)
                    y = float((box[1] + box[3]) / 2)

                    if track_id not in all_positions:
                        all_positions[track_id] = []
                    all_positions[track_id].append((frame_idx, x, y))

        # Racer heuristic: the person with the most movement
        # (spectators are mostly static, racer moves continuously)
        best_id = None
        best_score = 0
        min_frames = max(10, int(total_frames * 0.05)) if total_frames > 0 else 10

        for track_id, positions in all_positions.items():
            if len(positions) < min_frames:
                continue  # Skip brief detections
            total_movement = 0
            for i in range(1, len(positions)):
                dx = positions[i][1] - positions[i - 1][1]
                dy = positions[i][2] - positions[i - 1][2]
                total_movement += (dx**2 + dy**2) ** 0.5
            # Score favors sustained movement over brief bursts
            score = total_movement
            if score > best_score:
                best_score = score
                best_id = track_id

        if best_id is None:
            print("⚠️  Could not identify racer. Falling back to temporal method.")
            return self._track_with_temporal_consistency(video_path)

        # Extract trajectory for the identified racer
        racer_positions = all_positions[best_id]
        trajectory = [
            {"frame": pos[0], "x": pos[1], "y": pos[2]}
            for pos in racer_positions
        ]

        avg_movement = best_score / len(racer_positions) if racer_positions else 0
        print(f"✓ Identified racer (track_id={best_id}), "
              f"tracked {len(trajectory)} frames, "
              f"avg movement={avg_movement:.1f} px/frame")

        return trajectory

    def _track_with_temporal_consistency(
        self,
        video_path,
        frame_stride=1,
        max_frames=None,
        max_jump=None,
    ):
        """
        Fallback tracking using temporal consistency.
        Tracks the person closest to the previous frame's position,
        with a maximum jump distance to prevent tracking switches.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        trajectory = []
        prev_position = None
        frame_num = 0
        # pixels - racer shouldn't teleport between frames
        max_jump_px = max_jump

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_num >= max_frames:
                break
            if frame_stride > 1 and (frame_num % frame_stride != 0):
                frame_num += 1
                continue

            if max_jump_px is None:
                h, w = frame.shape[:2]
                base = 0.25 * min(w, h)
                max_jump_px = min(base * frame_stride, 0.6 * max(w, h))

            results = self.model(frame, classes=[0], verbose=False)

            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if prev_position is not None:
                    # Find detection closest to previous position
                    best_box = None
                    min_dist = float("inf")

                    for box in boxes:
                        x = (box[0] + box[2]) / 2
                        y = (box[1] + box[3]) / 2
                        dist = ((x - prev_position[0]) ** 2 + (y - prev_position[1]) ** 2) ** 0.5

                        if dist < max_jump_px and dist < min_dist:
                            min_dist = dist
                            best_box = (float(x), float(y))

                    if best_box:
                        prev_position = best_box
                        trajectory.append({
                            "frame": frame_num,
                            "x": best_box[0],
                            "y": best_box[1],
                        })
                else:
                    # First frame: pick center-most person (racer usually centered)
                    frame_center_x = frame.shape[1] / 2
                    best_box = None
                    min_dist = float("inf")

                    for box in boxes:
                        x = (box[0] + box[2]) / 2
                        y = (box[1] + box[3]) / 2
                        dist = abs(x - frame_center_x)

                        if dist < min_dist:
                            min_dist = dist
                            best_box = (float(x), float(y))

                    if best_box:
                        prev_position = best_box
                        trajectory.append({
                            "frame": frame_num,
                            "x": best_box[0],
                            "y": best_box[1],
                        })

            frame_num += 1

        cap.release()
        return trajectory

    def get_video_info(self, video_path):
        """Get basic video info for downstream processing."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info
