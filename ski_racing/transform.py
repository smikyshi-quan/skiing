"""
Perspective transformation module.
Converts 2D pixel coordinates to real-world 3D coordinates (bird's-eye view)
using homography based on known gate positions.

Includes camera motion compensation (Phase 2) and dynamic per-frame
scaling (Phase 4) for handling moving cameras in ski race footage.
"""
import cv2
import numpy as np


class CameraMotionCompensator:
    """
    Estimate and compensate for camera motion using detected gates as
    fixed-world anchor points.

    Since gates are physically fixed on the mountain, any movement of a
    gate's pixel position between frames is caused by camera panning/tilting.
    We compute the median gate displacement per frame relative to a baseline
    and subtract it from skier coordinates to get stabilized positions.
    """

    def __init__(self, baseline_gates, frame_gate_history, mode="translation"):
        """
        Args:
            baseline_gates: Dict {gate_id: (center_x, base_y)} — averaged
                           gate positions from temporal tracking (the "true" positions).
            frame_gate_history: Dict {frame_idx: {gate_id: (center_x, base_y)}} —
                               per-frame gate positions from TemporalGateTracker.
            mode: "translation" (default) or "affine" (rotation+translation+scale).
        """
        self.baseline = baseline_gates
        self.history = frame_gate_history
        self.mode = mode
        self.offsets = {}  # frame_idx -> (dx, dy)
        self.affine = {}   # frame_idx -> 2x3 affine matrix (baseline -> frame)
        self.affine_inv = {}  # frame_idx -> 2x3 inverse (frame -> baseline)

    def estimate_motion(self):
        """
        For each frame, compute camera motion relative to baseline gates.

        Translation mode:
            Median displacement of visible gates (dx, dy), with interpolation.
        Affine mode:
            Estimate affine transform (rotation + translation + scale) using
            matched gates. Missing frames fall back to nearest known transform.
        """
        if self.mode == "affine":
            self._estimate_affine_motion()
            return

        # Translation-only (default)
        raw_offsets = {}

        for frame_idx in sorted(self.history.keys()):
            frame_gates = self.history[frame_idx]
            dx_list = []
            dy_list = []

            for gate_id, (cx, by) in frame_gates.items():
                if gate_id in self.baseline:
                    bx, bby = self.baseline[gate_id]
                    dx_list.append(cx - bx)
                    dy_list.append(by - bby)

            if len(dx_list) >= 2:
                raw_offsets[frame_idx] = (
                    float(np.median(dx_list)),
                    float(np.median(dy_list)),
                )
            elif len(dx_list) == 1:
                raw_offsets[frame_idx] = (dx_list[0], dy_list[0])

        if not raw_offsets:
            self.offsets = {}
            return

        # Interpolate missing frames between known offsets
        all_frames = sorted(raw_offsets.keys())
        min_f, max_f = all_frames[0], all_frames[-1]

        for f in range(min_f, max_f + 1):
            if f in raw_offsets:
                self.offsets[f] = raw_offsets[f]
            else:
                # Find nearest known frames before and after
                prev_f = max(k for k in all_frames if k <= f) if any(k <= f for k in all_frames) else None
                next_f = min(k for k in all_frames if k >= f) if any(k >= f for k in all_frames) else None

                if prev_f is not None and next_f is not None and next_f != prev_f:
                    t = (f - prev_f) / (next_f - prev_f)
                    dx = raw_offsets[prev_f][0] + t * (raw_offsets[next_f][0] - raw_offsets[prev_f][0])
                    dy = raw_offsets[prev_f][1] + t * (raw_offsets[next_f][1] - raw_offsets[prev_f][1])
                    self.offsets[f] = (dx, dy)
                elif prev_f is not None:
                    self.offsets[f] = raw_offsets[prev_f]
                elif next_f is not None:
                    self.offsets[f] = raw_offsets[next_f]

        n_raw = len(raw_offsets)
        n_total = len(self.offsets)
        print(f"✓ Camera motion (translation): {n_raw} measured frames, {n_total} total (interpolated)")

        # Phase 2 diagnostic (Prof. feedback): Translation-only assumes the
        # camera slides without rotating. For broadcast/pan cameras this
        # introduces edge distortion — pixels on the left move faster than
        # the right during panning, biasing turn-radius calculations at
        # frame edges. Recommend affine mode for panning footage.
        if n_raw > 10:
            sorted_frames = sorted(self.offsets.keys())[:n_raw]
            dxs = [self.offsets[f][0] for f in sorted_frames]
            dys = [self.offsets[f][1] for f in sorted_frames]
            range_dx = max(dxs) - min(dxs) if dxs else 0
            # Large horizontal drift suggests panning camera
            if range_dx > 100:
                print(f"  ⚠️  Large horizontal camera drift detected ({range_dx:.0f}px). "
                      f"This suggests a panning camera. Translation-only compensation "
                      f"distorts turn radii at frame edges. Consider --camera-mode affine.")

    def _estimate_affine_motion(self):
        """
        Estimate rigid (Euclidean) motion per frame: rotation + translation ONLY.

        CRITICAL FIX: Previously used estimateAffinePartial2D which includes
        a scale component. In snowy/low-texture environments, the algorithm
        hallucinates zoom changes. When combined with DynamicScaleTransform,
        this creates a feedback loop:
          1. Affine "zoom" shrinks gate pixel distances
          2. DynamicScale sees smaller gate gaps → computes huge meters/pixel
          3. Small pixel movements → supersonic speeds (11,000 km/h)

        Now uses estimateAffinePartial2D but strips the scale component,
        keeping only rotation + translation (Euclidean rigid motion).
        Falls back to translation for frames with <3 gates.
        Interpolates missing frames between measured frames for continuity.
        """
        raw_affine = {}      # frame_idx -> 2x3 matrix (measured only)
        raw_affine_inv = {}  # frame_idx -> 2x3 inverse (measured only)
        n_scale_rejected = 0

        for frame_idx in sorted(self.history.keys()):
            frame_gates = self.history[frame_idx]
            src = []
            dst = []

            for gate_id, (cx, by) in frame_gates.items():
                if gate_id in self.baseline:
                    bx, bby = self.baseline[gate_id]
                    src.append([bx, bby])  # baseline
                    dst.append([cx, by])   # observed

            if len(src) >= 3:
                src_pts = np.array(src, dtype=np.float32)
                dst_pts = np.array(dst, dtype=np.float32)
                M, inliers = cv2.estimateAffinePartial2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                )
                if M is not None:
                    # Extract scale from the affine matrix.
                    # estimateAffinePartial2D returns:
                    #   [[s*cos(θ), -s*sin(θ), tx],
                    #    [s*sin(θ),  s*cos(θ), ty]]
                    # Scale = sqrt(M[0,0]^2 + M[1,0]^2)
                    scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))

                    # SAFETY: Strip scale component, keep only rotation+translation.
                    # This prevents the "affine zoom hallucination" in low-texture
                    # snow scenes that causes DynamicScale to explode.
                    if abs(scale - 1.0) > 0.01:
                        # Normalize to unit scale (Euclidean only)
                        M[0, 0] /= scale
                        M[0, 1] /= scale
                        M[1, 0] /= scale
                        M[1, 1] /= scale
                        n_scale_rejected += 1

                    raw_affine[frame_idx] = M
                    A = np.vstack([M, [0, 0, 1]])
                    try:
                        A_inv = np.linalg.inv(A)
                        raw_affine_inv[frame_idx] = A_inv[:2, :]
                    except np.linalg.LinAlgError:
                        pass
                    continue

            # Fallback: translation-only if we don't have enough gates
            if len(src) >= 1:
                dx = float(np.median([dst[i][0] - src[i][0] for i in range(len(src))]))
                dy = float(np.median([dst[i][1] - src[i][1] for i in range(len(src))]))
                M = np.array([[1.0, 0.0, dx],
                              [0.0, 1.0, dy]], dtype=np.float32)
                raw_affine[frame_idx] = M
                raw_affine_inv[frame_idx] = np.array([[1.0, 0.0, -dx],
                                                       [0.0, 1.0, -dy]], dtype=np.float32)

        if not raw_affine:
            return

        # Interpolate missing frames between measured frames
        all_frames = sorted(raw_affine.keys())
        min_f, max_f = all_frames[0], all_frames[-1]

        for f in range(min_f, max_f + 1):
            if f in raw_affine:
                self.affine[f] = raw_affine[f]
                if f in raw_affine_inv:
                    self.affine_inv[f] = raw_affine_inv[f]
            else:
                # Linear interpolation of affine matrix components
                prev_f = max(k for k in all_frames if k <= f) if any(k <= f for k in all_frames) else None
                next_f = min(k for k in all_frames if k >= f) if any(k >= f for k in all_frames) else None

                if prev_f is not None and next_f is not None and next_f != prev_f:
                    t = (f - prev_f) / (next_f - prev_f)
                    M_interp = (1 - t) * raw_affine[prev_f] + t * raw_affine[next_f]
                    self.affine[f] = M_interp
                    # Compute inverse of interpolated matrix
                    A = np.vstack([M_interp, [0, 0, 1]])
                    try:
                        A_inv = np.linalg.inv(A)
                        self.affine_inv[f] = A_inv[:2, :].astype(np.float32)
                    except np.linalg.LinAlgError:
                        # Fallback: interpolate inverses directly
                        if prev_f in raw_affine_inv and next_f in raw_affine_inv:
                            self.affine_inv[f] = (
                                (1 - t) * raw_affine_inv[prev_f] + t * raw_affine_inv[next_f]
                            )
                elif prev_f is not None:
                    self.affine[f] = raw_affine[prev_f]
                    if prev_f in raw_affine_inv:
                        self.affine_inv[f] = raw_affine_inv[prev_f]
                elif next_f is not None:
                    self.affine[f] = raw_affine[next_f]
                    if next_f in raw_affine_inv:
                        self.affine_inv[f] = raw_affine_inv[next_f]

        n_raw = len(raw_affine)
        n_total = len(self.affine)
        print(f"✓ Camera motion (rigid/Euclidean): {n_raw} measured, {n_total} total "
              f"(scale stripped from {n_scale_rejected} frames)")

    def _apply_affine_inv(self, M, x, y):
        """Apply a 2x3 inverse affine matrix to a point."""
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return (float(nx), float(ny))

    def _nearest_frame(self, frame_idx, frame_dict):
        """Find nearest frame key using bisect for O(log n) lookup."""
        if not frame_dict:
            return None
        import bisect
        if not hasattr(self, '_sorted_keys_cache') or self._sorted_keys_id != id(frame_dict):
            self._sorted_keys = sorted(frame_dict.keys())
            self._sorted_keys_id = id(frame_dict)
        keys = self._sorted_keys
        pos = bisect.bisect_left(keys, frame_idx)
        if pos == 0:
            return keys[0]
        if pos >= len(keys):
            return keys[-1]
        before, after = keys[pos - 1], keys[pos]
        return before if (frame_idx - before) <= (after - frame_idx) else after

    def stabilize_point(self, x, y, frame_idx):
        """
        Remove camera motion from a pixel coordinate.

        Returns:
            (x_stabilized, y_stabilized) as if camera never moved.
        """
        if self.mode == "affine":
            if frame_idx in self.affine_inv:
                return self._apply_affine_inv(self.affine_inv[frame_idx], x, y)
            if self.affine_inv:
                nearest = self._nearest_frame(frame_idx, self.affine_inv)
                if nearest is not None:
                    return self._apply_affine_inv(self.affine_inv[nearest], x, y)
            return (x, y)

        if frame_idx in self.offsets:
            dx, dy = self.offsets[frame_idx]
            return (x - dx, y - dy)

        # For frames outside known range, find nearest
        if self.offsets:
            nearest = self._nearest_frame(frame_idx, self.offsets)
            if nearest is not None:
                dx, dy = self.offsets[nearest]
                return (x - dx, y - dy)

        return (x, y)

    def stabilize_trajectory(self, trajectory_2d):
        """
        Apply camera stabilization to an entire 2D trajectory.

        Args:
            trajectory_2d: List of {"frame": int, "x": float, "y": float}

        Returns:
            New list with stabilized coordinates.
        """
        stabilized = []
        for pt in trajectory_2d:
            sx, sy = self.stabilize_point(pt["x"], pt["y"], pt["frame"])
            stabilized.append({"frame": pt["frame"], "x": sx, "y": sy})
        return stabilized


class DynamicScaleTransform:
    """
    Compute per-frame pixel-to-meter scaling from visible gate pairs.

    As the camera moves and zooms, the pixel distance between gates changes.
    By measuring gate-pair distances in each frame, we get a local scale
    that accounts for perspective depth variations.
    """

    def __init__(self, frame_gate_history, gate_spacing_m=12.0):
        """
        Args:
            frame_gate_history: Dict {frame_idx: {gate_id: (cx, by)}}
            gate_spacing_m: Real-world distance between consecutive gates.
        """
        self.history = frame_gate_history
        self.gate_spacing_m = gate_spacing_m
        self.frame_ppm = {}  # frame_idx -> ppm_y

    def compute_scales(self, global_ppm_y, one_gate_px_ref=None,
                        max_change_rate=0.10):
        """
        For each frame with >=2 visible gates, compute local ppm_y
        from the median gate-pair pixel distance.

        Includes scale hysteresis (damper): if the computed ppm changes
        by more than max_change_rate (default 10%) from the previous
        accepted value, the new value is REJECTED and the previous safe
        value is carried forward. This prevents the "affine zoom trap"
        where a bad stabilization frame causes gate distances to shrink,
        leading to explosive meters-per-pixel values.

        Args:
            global_ppm_y: Fallback ppm_y from static gate detection.
            one_gate_px_ref: Reference pixel distance for one gate interval.
            max_change_rate: Maximum allowed fractional change per frame (0.10 = 10%).
        """
        raw_ppm = {}  # Unfiltered per-frame ppm

        for frame_idx, frame_gates in self.history.items():
            if len(frame_gates) < 2:
                continue

            # Sort gates by base_y
            sorted_gates = sorted(frame_gates.values(), key=lambda g: g[1])
            ppm_estimates = []

            for i in range(len(sorted_gates) - 1):
                dy_px = sorted_gates[i + 1][1] - sorted_gates[i][1]
                if dy_px < 3.0:
                    continue  # Skip near-duplicate gates

                # Estimate how many real gate intervals this spans
                if one_gate_px_ref and one_gate_px_ref > 1.0:
                    n_gates = max(1, round(dy_px / one_gate_px_ref))
                else:
                    n_gates = 1

                dy_m = n_gates * self.gate_spacing_m
                if dy_m > 0:
                    ppm_estimates.append(dy_px / dy_m)

            if ppm_estimates:
                local_ppm = float(np.median(ppm_estimates))
                # Hard clamp to prevent extreme values (global bounds)
                if global_ppm_y:
                    local_ppm = max(0.3 * global_ppm_y, min(3.0 * global_ppm_y, local_ppm))
                raw_ppm[frame_idx] = local_ppm

        # ── Scale hysteresis: reject sudden jumps ──
        # Process frames in order; if ppm jumps by >max_change_rate from
        # the last accepted value, carry the previous value forward.
        n_rejected = 0
        prev_ppm = global_ppm_y  # Start from the static baseline

        for frame_idx in sorted(raw_ppm.keys()):
            candidate = raw_ppm[frame_idx]

            if prev_ppm is not None and prev_ppm > 0:
                change = abs(candidate - prev_ppm) / prev_ppm
                if change > max_change_rate:
                    # Reject: keep previous safe value
                    self.frame_ppm[frame_idx] = prev_ppm
                    n_rejected += 1
                    continue

            # Accept this frame's ppm
            self.frame_ppm[frame_idx] = candidate
            prev_ppm = candidate

        n_total = len(raw_ppm)
        n_accepted = n_total - n_rejected
        print(f"✓ Dynamic scale: {n_accepted}/{n_total} frames accepted "
              f"({n_rejected} rejected by hysteresis, "
              f"max_change_rate={max_change_rate:.0%})")

    def get_ppm(self, frame_idx, fallback_ppm):
        """
        Get pixels-per-meter for a specific frame.

        Falls back to nearest known frame or global fallback.
        """
        if frame_idx in self.frame_ppm:
            return self.frame_ppm[frame_idx]

        # Find nearest frame with known ppm using bisect for O(log n)
        if self.frame_ppm:
            if not hasattr(self, '_sorted_frames') or self._sorted_frames_len != len(self.frame_ppm):
                self._sorted_frames = sorted(self.frame_ppm.keys())
                self._sorted_frames_len = len(self.frame_ppm)
            import bisect
            keys = self._sorted_frames
            pos = bisect.bisect_left(keys, frame_idx)
            if pos == 0:
                nearest = keys[0]
            elif pos >= len(keys):
                nearest = keys[-1]
            else:
                before, after = keys[pos - 1], keys[pos]
                nearest = before if (frame_idx - before) <= (after - frame_idx) else after

            if abs(nearest - frame_idx) < 30:  # Within 1 second at 30fps
                return self.frame_ppm[nearest]

        return fallback_ppm


class HomographyTransform:
    """
    Transform 2D pixel coordinates to real-world meters using
    homography calculated from detected gate positions.
    """

    def __init__(self):
        self.H = None  # Homography matrix
        self.H_inv = None  # Inverse (for going 3D -> 2D)
        self.mode = "homography"
        self.y_map = None
        self.ppm_y = None
        self.ppm_bounds = None
        self.ppm_x = None
        self.gate_spacing_m = None
        self.centerline_fit = None
        self.camera_compensator = None  # Phase 2: CameraMotionCompensator
        self.dynamic_scale = None       # Phase 4: DynamicScaleTransform

    def calculate_from_gates(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate homography from detected gate positions.

        ⚠️  WARNING (Prof. feedback Phase 4): This computes a STATIC global
        homography from gates visible in the first frame. The perspective of
        the slope changes constantly as the skier moves down. A static
        homography is incorrect for the entire video.

        Prefer: --projection scale --stabilize (dynamic per-frame scaling).
        If you must use homography, it should be re-calculated per-frame
        using the 4 nearest gates (not yet implemented).

        Args:
            gates_2d: List of gate positions in pixels [(x, y), ...].
                      Should be sorted top-to-bottom (far to near).
            gate_spacing_m: Real-world distance between consecutive gates (meters).
                           Standard: ~12m for slalom, ~25-30m for GS.

        Returns:
            Homography matrix (3x3), or None if insufficient gates.
        """
        print("⚠️  Static homography: calculated ONCE from initial gate positions. "
              "This is incorrect for video where perspective changes as the "
              "skier moves down slope. Use --stabilize for dynamic scaling.")
        if len(gates_2d) < 4:
            print(f"⚠️  Only {len(gates_2d)} gates detected. Need at least 4 for homography.")
            if len(gates_2d) >= 2:
                print("   Using piecewise scale fallback from gate spacing.")
                self._calculate_scale_from_gates(gates_2d, gate_spacing_m)
            else:
                print("   Using identity transform as fallback.")
                self.H = np.eye(3)
                self.mode = "identity"
            return self.H

        # Create idealized 3D positions (bird's eye view)
        # Alternating left-right pattern typical of slalom/GS
        gates_3d = []
        for i in range(len(gates_2d)):
            # Slight lateral offset for alternating gates
            x_3d = (i % 2) * 3.0  # 3m lateral offset
            y_3d = i * gate_spacing_m
            gates_3d.append([x_3d, y_3d])

        # Use up to 8 gate pairs for better fit
        n_points = min(len(gates_2d), 8)
        pts_2d = np.float32(gates_2d[:n_points])
        pts_3d = np.float32(gates_3d[:n_points])

        # Calculate homography with RANSAC for robustness
        self.H, status = cv2.findHomography(pts_2d, pts_3d, cv2.RANSAC, 5.0)

        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            inliers = status.sum() if status is not None else n_points
            self.mode = "homography"
            print(f"✓ Homography calculated using {inliers}/{n_points} gate pairs")
        else:
            print("⚠️  Homography calculation failed. Falling back to piecewise scale.")
            if len(gates_2d) >= 2:
                self._calculate_scale_from_gates(gates_2d, gate_spacing_m)
            else:
                print("   Using identity transform as fallback.")
                self.H = np.eye(3)
                self.mode = "identity"

        return self.H

    def calculate_scale_from_gates(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate a piecewise scale mapping from gate positions.
        This is a pseudo-3D projection that avoids global homography.
        """
        if len(gates_2d) < 2:
            print(f"⚠️  Only {len(gates_2d)} gates detected. Need at least 2 for scale mapping.")
            print("   Using identity transform as fallback.")
            self.H = np.eye(3)
            self.mode = "identity"
            return self.H

        self._calculate_scale_from_gates(gates_2d, gate_spacing_m)
        return self.H

    def transform_point(self, point_2d, frame_idx=None):
        """
        Transform a single 2D pixel coordinate to 3D world coordinates.

        Args:
            point_2d: (x, y) in pixels.
            frame_idx: Optional frame index for dynamic scale (Phase 4).

        Returns:
            (x, y) in meters (bird's-eye view).
        """
        if self.mode == "homography":
            if self.H is None:
                raise ValueError("Must calculate homography first")

            pt = np.array([[point_2d[0], point_2d[1]]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt, self.H)
            return (float(transformed[0][0][0]), float(transformed[0][0][1]))

        if self.mode == "scale":
            if not self.y_map or len(self.y_map) < 2:
                return (float(point_2d[0]), float(point_2d[1]))

            y_px = float(point_2d[1])
            # Find enclosing gate segment
            idx = 0
            for i in range(len(self.y_map) - 1):
                if self.y_map[i]["y_px"] <= y_px <= self.y_map[i + 1]["y_px"]:
                    idx = i
                    break
                if y_px < self.y_map[0]["y_px"]:
                    idx = 0
                    break
                if y_px > self.y_map[-1]["y_px"]:
                    idx = len(self.y_map) - 2
                    break

            g0 = self.y_map[idx]
            g1 = self.y_map[idx + 1]
            y0, y1 = g0["y_px"], g1["y_px"]
            # Real meter span of this segment (accounts for skipped gates)
            segment_m = g1["y_m"] - g0["y_m"]
            if segment_m < 1e-6:
                segment_m = self.gate_spacing_m

            if abs(y1 - y0) < 1e-6:
                ppm_y = self.ppm_y or 1.0
                t = 0.0
            else:
                ppm_y = (y1 - y0) / segment_m
                t = (y_px - y0) / (y1 - y0)

            # Phase 4: Use dynamic per-frame scale if available
            if self.dynamic_scale is not None and frame_idx is not None:
                dyn_ppm = self.dynamic_scale.get_ppm(frame_idx, None)
                if dyn_ppm is not None:
                    ppm_y = dyn_ppm

            # Clamp ppm to avoid extreme scale from near-duplicate gates
            if self.ppm_y is not None:
                low, high = self.ppm_bounds if self.ppm_bounds else (0.5 * self.ppm_y, 2.0 * self.ppm_y)
                if ppm_y < low or ppm_y > high:
                    ppm_y = self.ppm_y

            y_m = g0["y_m"] + t * segment_m
            if self.centerline_fit is not None:
                a, b = self.centerline_fit
                x_ref = a * y_px + b
            else:
                x_ref = (g0["x_px"] + g1["x_px"]) / 2.0
            # Use local ppm_y for X as well (isotropic at each depth)
            x_m = (float(point_2d[0]) - x_ref) / ppm_y
            return (x_m, y_m)

        # Identity fallback
        return (float(point_2d[0]), float(point_2d[1]))

    def transform_trajectory(self, trajectory_2d, stabilize=False,
                              max_jump_m=5.0, fps=30.0):
        """
        Transform an entire 2D trajectory to 3D coordinates.

        Includes a physical jump guard: if any point-to-point 3D distance
        exceeds max_jump_m (default 5m per frame ≈ 540 km/h at 30fps),
        the point is replaced by linear interpolation from its neighbors.
        This prevents a single bad scale frame from corrupting the trajectory.

        Args:
            trajectory_2d: List of {"frame": int, "x": float, "y": float}.
            stabilize: If True, apply camera motion compensation first.
            max_jump_m: Maximum allowed 3D jump per frame (meters).
            fps: Video frame rate (for scaling max_jump with frame gaps).

        Returns:
            List of {"frame": int, "x": float, "y": float} in meters.
        """
        if self.H is None:
            raise ValueError("Must calculate homography first")

        # Phase 2: Apply camera stabilization if enabled
        if stabilize and self.camera_compensator is not None:
            working_trajectory = self.camera_compensator.stabilize_trajectory(trajectory_2d)
        else:
            working_trajectory = trajectory_2d

        trajectory_3d = []
        for point in working_trajectory:
            frame_idx = point.get("frame")
            pt_3d = self.transform_point([point["x"], point["y"]], frame_idx=frame_idx)
            trajectory_3d.append({
                "frame": point["frame"],
                "x": pt_3d[0],
                "y": pt_3d[1],
            })

        # ── Physical jump guard ──
        # If a 3D point-to-point distance exceeds max_jump_m (scaled by
        # frame gap), replace it with linear interpolation from neighbors.
        # This catches the "109m jump" scenario where a single bad scale
        # frame causes a physically impossible displacement.
        n_clamped = 0
        for i in range(1, len(trajectory_3d) - 1):
            prev_pt = trajectory_3d[i - 1]
            curr_pt = trajectory_3d[i]
            next_pt = trajectory_3d[i + 1]

            # Scale max jump by frame gap (if frames aren't consecutive)
            frame_gap = max(1, curr_pt["frame"] - prev_pt["frame"])
            allowed_jump = max_jump_m * frame_gap

            dx = curr_pt["x"] - prev_pt["x"]
            dy = curr_pt["y"] - prev_pt["y"]
            jump = (dx**2 + dy**2) ** 0.5

            if jump > allowed_jump:
                # Replace with interpolated position from neighbors
                f_prev = prev_pt["frame"]
                f_curr = curr_pt["frame"]
                f_next = next_pt["frame"]
                if f_next > f_prev:
                    t = (f_curr - f_prev) / (f_next - f_prev)
                else:
                    t = 0.5
                trajectory_3d[i]["x"] = prev_pt["x"] + t * (next_pt["x"] - prev_pt["x"])
                trajectory_3d[i]["y"] = prev_pt["y"] + t * (next_pt["y"] - prev_pt["y"])
                n_clamped += 1

        if n_clamped > 0:
            print(f"  ⚠️  Jump guard: {n_clamped} points had >5m jumps and were interpolated")

        return trajectory_3d

    def get_reprojection_error(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate reprojection error to assess homography quality.
        Lower is better. >10 pixels means the transform is suspect.

        Returns:
            Mean reprojection error in pixels.
        """
        if self.H is None or self.H_inv is None:
            return float("inf")

        errors = []
        for i, gate in enumerate(gates_2d):
            # Forward transform: 2D -> 3D
            pt_3d = self.transform_point(gate)

            # Inverse: 3D -> 2D
            pt_3d_arr = np.array([[pt_3d[0], pt_3d[1]]], dtype=np.float32).reshape(-1, 1, 2)
            reprojected = cv2.perspectiveTransform(pt_3d_arr, self.H_inv)
            rx, ry = reprojected[0][0]

            error = ((rx - gate[0]) ** 2 + (ry - gate[1]) ** 2) ** 0.5
            errors.append(error)

        mean_error = np.mean(errors)
        print(f"Reprojection error: {mean_error:.2f} pixels (target: <10)")
        return mean_error

    def _calculate_scale_from_gates(self, gates_2d, gate_spacing_m):
        """
        Piecewise linear scale mapping using gate spacing.
        Maps y using gate spacing and scales x using local y-scale.

        Key fix: we estimate how many real gates are between each pair of
        detected gates using the minimum-gap heuristic, so that skipped
        gates don't compress the meter scale.
        """
        gates_sorted = sorted(gates_2d, key=lambda g: g[1])
        self.gate_spacing_m = float(gate_spacing_m)

        # --- Step 1: estimate real gate counts between detected gates ---
        # Compute all Y-pixel gaps between consecutive detected gates
        y_gaps = []
        for i in range(1, len(gates_sorted)):
            dy = gates_sorted[i][1] - gates_sorted[i - 1][1]
            if dy > 1e-3:
                y_gaps.append(dy)

        if y_gaps:
            # The smallest gap most likely represents ONE real gate interval.
            # Use a low percentile (not absolute min) for robustness against
            # near-duplicate detections that survived clustering.
            y_gaps_sorted = sorted(y_gaps)
            # Use 20th percentile as "one gate" reference
            ref_idx = max(0, int(0.2 * (len(y_gaps_sorted) - 1)))
            one_gate_px = y_gaps_sorted[ref_idx]
        else:
            one_gate_px = None

        # --- Step 2: build y_map with corrected meter positions ---
        self.y_map = []
        cumulative_m = 0.0
        self.y_map.append({
            "y_px": float(gates_sorted[0][1]),
            "y_m": 0.0,
            "x_px": float(gates_sorted[0][0]),
            "n_gates_from_prev": 0,
        })

        for i in range(1, len(gates_sorted)):
            dy_px = gates_sorted[i][1] - gates_sorted[i - 1][1]
            if one_gate_px is not None and one_gate_px > 1e-3:
                # Estimate how many real gates this gap spans
                n_gates = max(1, round(dy_px / one_gate_px))
            else:
                n_gates = 1
            cumulative_m += n_gates * self.gate_spacing_m
            self.y_map.append({
                "y_px": float(gates_sorted[i][1]),
                "y_m": cumulative_m,
                "x_px": float(gates_sorted[i][0]),
                "n_gates_from_prev": n_gates,
            })

        # --- Step 3: compute ppm_y from corrected spacing ---
        ppm_vals = []
        for i in range(1, len(self.y_map)):
            dy_px = self.y_map[i]["y_px"] - self.y_map[i - 1]["y_px"]
            real_m = self.y_map[i]["y_m"] - self.y_map[i - 1]["y_m"]
            if dy_px > 1e-3 and real_m > 1e-3:
                ppm_vals.append(dy_px / real_m)
        if ppm_vals:
            self.ppm_y = float(np.median(ppm_vals))
            ppm_vals_sorted = sorted(ppm_vals)
            p10 = ppm_vals_sorted[max(0, int(0.1 * (len(ppm_vals_sorted) - 1)))]
            p90 = ppm_vals_sorted[min(len(ppm_vals_sorted) - 1, int(0.9 * (len(ppm_vals_sorted) - 1)))]
            self.ppm_bounds = (p10, p90)
        else:
            self.ppm_y = None
            self.ppm_bounds = None

        # --- Step 4: ppm_x = local ppm_y (isotropic at each depth) ---
        # In perspective projection, horizontal and vertical scale are the
        # same at any given depth. The old lateral-offset estimation was
        # fragile when detected gates weren't truly alternating.
        self.ppm_x = None  # will use local ppm_y in transform_point

        self.mode = "scale"
        self.H = np.eye(3)

        # Fit a simple centerline x = a*y + b to avoid lateral jumps
        try:
            ys = np.array([g["y_px"] for g in self.y_map], dtype=float)
            xs = np.array([g["x_px"] for g in self.y_map], dtype=float)
            if len(xs) >= 2:
                a, b = np.polyfit(ys, xs, 1)
                self.centerline_fit = (float(a), float(b))
            else:
                self.centerline_fit = None
        except Exception:
            self.centerline_fit = None

        # Print diagnostics
        total_gates_est = sum(g["n_gates_from_prev"] for g in self.y_map)
        ppm_text = f"{self.ppm_y:.2f}" if self.ppm_y else "n/a"
        ref_text = f"{one_gate_px:.1f}" if one_gate_px else "n/a"
        print(f"✓ Scale mapping: {len(self.y_map)} detected gates, "
              f"~{total_gates_est} estimated real gates")
        print(f"  one-gate reference: {ref_text} px, "
              f"ppm_y ≈ {ppm_text} px/m, "
              f"course ≈ {cumulative_m:.0f}m")
