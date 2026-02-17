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


class TrajectoryOutlierFilter:
    """
    Robust pre-smoothing outlier filter for 2D trajectories.

    Uses a sliding-window median and MAD test. Flagged points are replaced
    by interpolation between nearest inlier neighbors while preserving the
    original detections for debugging.
    """

    def __init__(self, window=5, mad_threshold=3.0, min_mad_px=1.0):
        self.window = max(3, int(window))
        if self.window % 2 == 0:
            self.window += 1
        self.mad_threshold = float(mad_threshold)
        self.min_mad_px = float(min_mad_px)

    def filter(self, trajectory_2d):
        """
        Args:
            trajectory_2d: List of {"frame", "x", "y", ...}

        Returns:
            (filtered_trajectory, diagnostics)
        """
        n = len(trajectory_2d)
        if n == 0:
            return trajectory_2d, {"outlier_count": 0, "outlier_frames": []}

        filtered = [dict(pt) for pt in trajectory_2d]
        half = self.window // 2
        is_outlier = [False] * n

        xs = np.array([float(pt["x"]) for pt in trajectory_2d], dtype=np.float64)
        ys = np.array([float(pt["y"]) for pt in trajectory_2d], dtype=np.float64)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)

            win_x = xs[lo:hi]
            win_y = ys[lo:hi]
            if len(win_x) < 3:
                continue

            med_x = float(np.median(win_x))
            med_y = float(np.median(win_y))
            mad_x = float(np.median(np.abs(win_x - med_x)))
            mad_y = float(np.median(np.abs(win_y - med_y)))

            limit_x = self.mad_threshold * max(self.min_mad_px, mad_x)
            limit_y = self.mad_threshold * max(self.min_mad_px, mad_y)

            dx = abs(float(xs[i]) - med_x)
            dy = abs(float(ys[i]) - med_y)
            if dx > limit_x or dy > limit_y:
                is_outlier[i] = True

        outlier_frames = []
        for i, pt in enumerate(filtered):
            raw_x = float(pt["x"])
            raw_y = float(pt["y"])
            pt["raw_x"] = raw_x
            pt["raw_y"] = raw_y
            pt["is_outlier"] = bool(is_outlier[i])
            if is_outlier[i]:
                outlier_frames.append(int(pt["frame"]))

        inlier_indices = [i for i in range(n) if not is_outlier[i]]
        if not inlier_indices:
            return filtered, {
                "outlier_count": int(n),
                "outlier_frames": outlier_frames,
            }

        for i in range(n):
            if not is_outlier[i]:
                continue

            prev_idx = None
            next_idx = None

            j = i - 1
            while j >= 0:
                if not is_outlier[j]:
                    prev_idx = j
                    break
                j -= 1

            j = i + 1
            while j < n:
                if not is_outlier[j]:
                    next_idx = j
                    break
                j += 1

            if prev_idx is not None and next_idx is not None:
                f0 = float(filtered[prev_idx]["frame"])
                f1 = float(filtered[next_idx]["frame"])
                fk = float(filtered[i]["frame"])
                t = (fk - f0) / (f1 - f0) if f1 > f0 else 0.5
                x0, y0 = float(filtered[prev_idx]["x"]), float(filtered[prev_idx]["y"])
                x1, y1 = float(filtered[next_idx]["x"]), float(filtered[next_idx]["y"])
                filtered[i]["x"] = float(x0 + t * (x1 - x0))
                filtered[i]["y"] = float(y0 + t * (y1 - y0))
                c0 = float(filtered[prev_idx].get("confidence", 0.5))
                c1 = float(filtered[next_idx].get("confidence", 0.5))
                filtered[i]["confidence"] = float(0.5 * (c0 + c1))
            elif prev_idx is not None:
                filtered[i]["x"] = float(filtered[prev_idx]["x"])
                filtered[i]["y"] = float(filtered[prev_idx]["y"])
                filtered[i]["confidence"] = float(filtered[prev_idx].get("confidence", 0.5))
            elif next_idx is not None:
                filtered[i]["x"] = float(filtered[next_idx]["x"])
                filtered[i]["y"] = float(filtered[next_idx]["y"])
                filtered[i]["confidence"] = float(filtered[next_idx].get("confidence", 0.5))

        return filtered, {
            "outlier_count": int(len(outlier_frames)),
            "outlier_frames": outlier_frames,
        }


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
                 discipline=None, innovation_gate=9.0,
                 low_conf_threshold=0.5, low_conf_r_multiplier=2.5):
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
            innovation_gate: Mahalanobis gate threshold for innovation
                             (squared distance). 9.0 ≈ 3σ for 2D.
            low_conf_threshold: Detection confidence threshold where
                                measurement noise starts increasing.
            low_conf_r_multiplier: Maximum multiplier applied to
                                   measurement noise std-dev when confidence
                                   is near zero.
        """
        self.dt = 1.0 / fps
        self.discipline = discipline or "slalom"

        defaults = self.DISCIPLINE_DEFAULTS.get(self.discipline, self.DISCIPLINE_DEFAULTS["slalom"])
        self.process_noise = process_noise if process_noise is not None else defaults["process_noise"]
        self.measurement_noise = measurement_noise if measurement_noise is not None else defaults["measurement_noise"]
        self.innovation_gate = float(innovation_gate) if innovation_gate is not None else None
        self.low_conf_threshold = float(low_conf_threshold)
        self.low_conf_r_multiplier = float(max(1.0, low_conf_r_multiplier))

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
        gate_rejected = 0
        gate_considered = 0

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
                # Update with measurement (Mahalanobis innovation gating)
                y_innov = z - H @ state
                det_conf = float(pt.get("confidence", 1.0))
                conf_for_scale = float(np.clip(det_conf, 0.0, 1.0))
                r_scale = 1.0
                if conf_for_scale < self.low_conf_threshold and self.low_conf_threshold > 1e-6:
                    frac = (self.low_conf_threshold - conf_for_scale) / self.low_conf_threshold
                    r_scale = 1.0 + frac * (self.low_conf_r_multiplier - 1.0)
                R_eff = R * (r_scale ** 2)
                S = H @ P @ H.T + R_eff
                use_update = True
                gate_considered += 1

                if self.innovation_gate is not None:
                    try:
                        S_inv = np.linalg.inv(S)
                        mahal_sq = float(y_innov.T @ S_inv @ y_innov)
                        if mahal_sq > self.innovation_gate:
                            use_update = False
                            gate_rejected += 1
                    except np.linalg.LinAlgError:
                        use_update = False

                if use_update:
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

        if gate_rejected > 0:
            print(f"  ✓ Innovation gating: skipped {gate_rejected}/{gate_considered} "
                  f"updates (mahal_sq > {self.innovation_gate:.1f})")

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
        self._tracker_cfg = self._resolve_tracker_config_path()
        self._last_tracking_stats = {}

    @staticmethod
    def _resolve_tracker_config_path():
        """Resolve local ByteTrack config path with tuned thresholds."""
        project_root = Path(__file__).resolve().parent.parent
        cfg_path = project_root / "configs" / "bytetrack_tuned.yaml"
        if cfg_path.exists():
            return str(cfg_path)
        return "bytetrack.yaml"

    def get_last_tracking_stats(self):
        return dict(self._last_tracking_stats) if self._last_tracking_stats else {}

    def track_video(
        self,
        video_path,
        method="bytetrack",
        frame_stride=1,
        max_frames=None,
        max_jump=None,
        conf=0.25,
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
                trajectory = self._track_with_bytetrack_reassociate(
                    video_path,
                    conf=conf,
                    max_jump=max_jump,
                )
                return trajectory
            except ModuleNotFoundError as e:
                print(f"⚠️  ByteTrack dependency missing ({e}). Falling back to temporal tracking.")
                trajectory = self._track_with_temporal_consistency(
                    video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                    max_jump=max_jump,
                    conf=conf,
                )
                self._last_tracking_stats["selected_method"] = "temporal"
                self._last_tracking_stats["failure_reason"] = f"ModuleNotFoundError: {e}"
                return trajectory
            except Exception as e:
                print(f"⚠️  ByteTrack failed ({e}). Falling back to temporal tracking.")
                trajectory = self._track_with_temporal_consistency(
                    video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                    max_jump=max_jump,
                    conf=conf,
                )
                self._last_tracking_stats["selected_method"] = "temporal"
                self._last_tracking_stats["failure_reason"] = f"Exception: {e}"
                return trajectory
        else:
            trajectory = self._track_with_temporal_consistency(
                video_path,
                frame_stride=frame_stride,
                max_frames=max_frames,
                max_jump=max_jump,
                conf=conf,
            )
            self._last_tracking_stats["selected_method"] = "temporal"
            return trajectory

    def _track_with_bytetrack_reassociate(self, video_path, conf=0.25, max_jump=None,
                                          max_gap=30, reacquire_gap=15):
        """
        Use YOLOv8's built-in ByteTrack detections but reassociate
        targets frame-to-frame to avoid ID switches truncating the run.
        """
        trajectory = []

        # Run tracking with persistence
        results = self.model.track(
            source=video_path,
            classes=[0],  # person class only
            persist=True,
            tracker=self._tracker_cfg,
            conf=conf,
            verbose=False,
            stream=True,
        )

        last_pos = None
        last_vel = (0.0, 0.0)
        last_frame = None
        last_area = None
        last_track_id = None
        missing = 0
        track_id_switches = 0
        track_id_changes_observed = 0
        observed_frames = 0

        for frame_idx, result in enumerate(results):
            observed_frames += 1
            if result.boxes is None or len(result.boxes) == 0:
                missing += 1
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

            h, w = result.orig_shape[:2] if hasattr(result, "orig_shape") else (None, None)
            if w is None or h is None:
                # Fallback when orig_shape is missing
                h, w = 720, 1280

            detections = []
            for i, box in enumerate(boxes):
                x = float((box[0] + box[2]) / 2)
                y = float(box[3])
                area = float((box[2] - box[0]) * (box[3] - box[1]))
                c = float(confs[i]) if confs is not None else 0.5
                det_id = int(ids[i]) if ids is not None and not np.isnan(ids[i]) else None
                detections.append({
                    "x": x,
                    "y": y,
                    "area": area,
                    "confidence": c,
                    "track_id": det_id,
                })

            if not detections:
                missing += 1
                continue

            # Compute dynamic max jump
            if max_jump is None:
                base = 0.35 * min(w, h)
                max_jump_px = min(base, 0.75 * max(w, h))
            else:
                max_jump_px = float(max_jump)

            chosen = None
            if last_pos is None:
                # First frame: choose detection closest to center
                cx, cy = w / 2.0, h / 2.0
                best = None
                best_dist = float("inf")
                for det in detections:
                    dist = ((det["x"] - cx) ** 2 + (det["y"] - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best = det
                chosen = best
            else:
                gap = max(1, frame_idx - (last_frame if last_frame is not None else frame_idx))
                pred_x = last_pos[0] + last_vel[0] * gap
                pred_y = last_pos[1] + last_vel[1] * gap

                if last_track_id is not None:
                    for det in detections:
                        if det.get("track_id") != last_track_id:
                            continue
                        dist_same = ((det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2) ** 0.5
                        if dist_same <= max_jump_px * gap:
                            chosen = det
                            break

                if chosen is not None:
                    pass
                else:
                    best = None
                    best_dist = float("inf")
                    for det in detections:
                        dist = ((det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best = det

                    if best is not None and best_dist <= max_jump_px * gap:
                        chosen = best
                    elif gap > reacquire_gap:
                        # Reacquire: choose most prominent detection
                        best = None
                        best_score = -float("inf")
                        for det in detections:
                            score = (det["area"] ** 0.5) * det["confidence"]  # area^0.5 * conf
                            if score > best_score:
                                best_score = score
                                best = det
                        if best is not None:
                            within_pred = ((best["x"] - pred_x) ** 2 + (best["y"] - pred_y) ** 2) ** 0.5 <= 200.0
                            if last_area is not None and last_area > 1e-6:
                                area_ratio = best["area"] / last_area
                                within_size = 0.5 <= area_ratio <= 1.5
                            else:
                                within_size = True
                            if within_pred and within_size:
                                chosen = best
                    else:
                        missing += 1
                        if missing > max_gap:
                            last_pos = None
                            last_vel = (0.0, 0.0)
                            last_frame = None
                            last_area = None
                            last_track_id = None
                        continue

            if chosen is None:
                missing += 1
                continue

            x, y = chosen["x"], chosen["y"]
            gap_to_prev = max(1, frame_idx - last_frame) if last_frame is not None else 1
            if last_pos is not None and last_frame is not None:
                gap = max(1, frame_idx - last_frame)
                last_vel = ((x - last_pos[0]) / gap, (y - last_pos[1]) / gap)
            current_track_id = chosen.get("track_id")
            if (last_track_id is not None and current_track_id is not None
                    and current_track_id != last_track_id):
                track_id_changes_observed += 1
                jump_px = ((x - last_pos[0]) ** 2 + (y - last_pos[1]) ** 2) ** 0.5 if last_pos is not None else 0.0
                # Count only continuity-breaking switches.
                if gap_to_prev <= 2 and jump_px > 500.0:
                    track_id_switches += 1
            last_pos = (x, y)
            last_frame = frame_idx
            last_area = float(chosen["area"])
            last_track_id = current_track_id
            missing = 0

            trajectory.append({
                "frame": frame_idx,
                "x": x,
                "y": y,
                "confidence": float(chosen["confidence"]),
                "track_id": current_track_id,
                "bbox_area": float(chosen["area"]),
            })

        print(f"✓ ByteTrack reassociation: tracked {len(trajectory)} frames")

        # If coverage is too low, fall back to temporal tracking
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            total_frames = 0

        bytetrack_valid_frames = len(trajectory)
        coverage = (bytetrack_valid_frames / total_frames) if total_frames > 0 else 0.0
        self._last_tracking_stats = {
            "method_requested": "bytetrack",
            "selected_method": "bytetrack",
            "total_frames": int(total_frames) if total_frames > 0 else int(observed_frames),
            "frames_with_valid_track": int(bytetrack_valid_frames),
            "bytetrack_coverage": float(coverage),
            "frames_fallback_used": 0,
            "fallback_used": False,
            "track_id_switches": int(track_id_switches),
            "track_id_changes_observed": int(track_id_changes_observed),
            "tracker_config": self._tracker_cfg,
        }

        if total_frames > 0 and coverage < 0.4:
            print(f"⚠️  Low ByteTrack coverage ({coverage:.0%}). Falling back to temporal tracking.")
            fallback_trajectory = self._track_with_temporal_consistency(
                video_path,
                frame_stride=1,
                max_frames=None,
                max_jump=max_jump,
                conf=conf,
                update_stats=False,
            )
            self._last_tracking_stats["selected_method"] = "temporal_fallback"
            self._last_tracking_stats["frames_fallback_used"] = int(len(fallback_trajectory))
            self._last_tracking_stats["fallback_used"] = True
            return fallback_trajectory

        return trajectory

    def _track_with_temporal_consistency(
        self,
        video_path,
        frame_stride=1,
        max_frames=None,
        max_jump=None,
        conf=0.25,
        update_stats=True,
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
                base = 0.35 * min(w, h)
                max_jump_px = min(base * frame_stride, 0.75 * max(w, h))

            results = self.model(frame, classes=[0], conf=conf, verbose=False)

            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None

                if prev_position is not None:
                    # Find detection closest to previous position
                    best_box = None
                    min_dist = float("inf")

                    for i, box in enumerate(boxes):
                        x = (box[0] + box[2]) / 2
                        y = box[3]
                        dist = ((x - prev_position[0]) ** 2 + (y - prev_position[1]) ** 2) ** 0.5

                        if dist < max_jump_px and dist < min_dist:
                            min_dist = dist
                            c = float(confs[i]) if confs is not None else 0.5
                            area = float((box[2] - box[0]) * (box[3] - box[1]))
                            best_box = (float(x), float(y), c, area)

                    if best_box:
                        prev_position = (best_box[0], best_box[1])
                        trajectory.append({
                            "frame": frame_num,
                            "x": best_box[0],
                            "y": best_box[1],
                            "confidence": float(best_box[2]),
                            "track_id": None,
                            "bbox_area": float(best_box[3]),
                        })
                else:
                    # First frame: pick center-most person (racer usually centered)
                    frame_center_x = frame.shape[1] / 2
                    best_box = None
                    min_dist = float("inf")

                    for i, box in enumerate(boxes):
                        x = (box[0] + box[2]) / 2
                        y = box[3]
                        dist = abs(x - frame_center_x)

                        if dist < min_dist:
                            min_dist = dist
                            c = float(confs[i]) if confs is not None else 0.5
                            area = float((box[2] - box[0]) * (box[3] - box[1]))
                            best_box = (float(x), float(y), c, area)

                    if best_box:
                        prev_position = (best_box[0], best_box[1])
                        trajectory.append({
                            "frame": frame_num,
                            "x": best_box[0],
                            "y": best_box[1],
                            "confidence": float(best_box[2]),
                            "track_id": None,
                            "bbox_area": float(best_box[3]),
                        })

            frame_num += 1

        cap.release()
        if update_stats:
            self._last_tracking_stats = {
                "method_requested": "temporal",
                "selected_method": "temporal",
                "total_frames": int(frame_num),
                "frames_with_valid_track": int(len(trajectory)),
                "bytetrack_coverage": 0.0,
                "frames_fallback_used": int(frame_num),
                "fallback_used": False,
                "track_id_switches": 0,
                "track_id_changes_observed": 0,
            }
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
