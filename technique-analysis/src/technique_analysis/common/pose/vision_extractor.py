"""Apple Vision 3D pose extractor — spike replacement for MediaPipe.

Uses two Apple Vision requests on the same handler call:
  - VNDetectHumanBodyPoseRequest      → 2D normalised image coordinates
  - VNDetectHumanBodyPose3DRequest    → 3D metric body-space coordinates

Advantages over MediaPipe PoseLandmarker:
  - Dispatched to ANE (Apple Neural Engine) via CoreML automatically.
  - No separate model file download; ships with macOS 14+.
  - Returns body height in metres — useful calibration signal.
  - Single API call returns both 2D and 3D.

Limitations vs MediaPipe:
  - Only 17 joints (no feet, no face landmarks, no fingertips).
  - Cannot be fine-tuned on skiing data.
  - 2D joints use bottom-left origin → we flip y here.
  - 3D joints use y-UP convention → we negate y here to match
    MediaPipe's y-DOWN world_landmarks convention.

Coordinate mappings (Vision joint → MediaPipe index):
  0  nose/head  ← Nose (head_joint)
  11 L_shoulder ← LeftShoulder
  12 R_shoulder ← RightShoulder
  13 L_elbow    ← LeftElbow
  14 R_elbow    ← RightElbow
  15 L_wrist    ← LeftWrist
  16 R_wrist    ← RightWrist
  23 L_hip      ← LeftHip
  24 R_hip      ← RightHip
  25 L_knee     ← LeftKnee
  26 R_knee     ← RightKnee
  27 L_ankle    ← LeftAnkle
  28 R_ankle    ← RightAnkle
  (slots without a Vision equivalent get visibility=0.0)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from technique_analysis.common.contracts.models import FramePose, PoseLandmark
from technique_analysis.common.pose.person_detector import PersonDetector
from technique_analysis.common.pose.tracker import PersonTracker

# Key joints for confidence scoring (same indices as MediaPipe extractor)
_CONFIDENCE_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28]

# Total landmark slots — keeps the same 33-element list as MediaPipe
_N_LANDMARKS = 33

# Minimum macOS version for VNDetectHumanBodyPose3DRequest
_MIN_MACOS = (14, 0)


# ---------------------------------------------------------------------------
# Joint index maps
# ---------------------------------------------------------------------------

# 2D Vision joint key constant name → MediaPipe landmark index
_JOINT_2D_TO_IDX: dict[str, int] = {
    "VNHumanBodyPoseObservationJointNameNose":          0,
    "VNHumanBodyPoseObservationJointNameLeftShoulder":  11,
    "VNHumanBodyPoseObservationJointNameRightShoulder": 12,
    "VNHumanBodyPoseObservationJointNameLeftElbow":     13,
    "VNHumanBodyPoseObservationJointNameRightElbow":    14,
    "VNHumanBodyPoseObservationJointNameLeftWrist":     15,
    "VNHumanBodyPoseObservationJointNameRightWrist":    16,
    "VNHumanBodyPoseObservationJointNameLeftHip":       23,
    "VNHumanBodyPoseObservationJointNameRightHip":      24,
    "VNHumanBodyPoseObservationJointNameLeftKnee":      25,
    "VNHumanBodyPoseObservationJointNameRightKnee":     26,
    "VNHumanBodyPoseObservationJointNameLeftAnkle":     27,
    "VNHumanBodyPoseObservationJointNameRightAnkle":    28,
}

# 3D Vision joint key constant name → MediaPipe landmark index
_JOINT_3D_TO_IDX: dict[str, int] = {
    "VNHumanBodyPose3DObservationJointNameCenterHead":    0,
    "VNHumanBodyPose3DObservationJointNameLeftShoulder":  11,
    "VNHumanBodyPose3DObservationJointNameRightShoulder": 12,
    "VNHumanBodyPose3DObservationJointNameLeftElbow":     13,
    "VNHumanBodyPose3DObservationJointNameRightElbow":    14,
    "VNHumanBodyPose3DObservationJointNameLeftWrist":     15,
    "VNHumanBodyPose3DObservationJointNameRightWrist":    16,
    "VNHumanBodyPose3DObservationJointNameLeftHip":       23,
    "VNHumanBodyPose3DObservationJointNameRightHip":      24,
    "VNHumanBodyPose3DObservationJointNameLeftKnee":      25,
    "VNHumanBodyPose3DObservationJointNameRightKnee":     26,
    "VNHumanBodyPose3DObservationJointNameLeftAnkle":     27,
    "VNHumanBodyPose3DObservationJointNameRightAnkle":    28,
}

# Regex for extracting float values from Vision's simd_float4x4 repr
_MATRIX_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _check_macos_version() -> bool:
    """Return True if macOS ≥ 14.0 (required for 3D pose)."""
    import platform
    ver = platform.mac_ver()[0]
    if not ver:
        return False
    parts = tuple(int(x) for x in ver.split(".")[:2])
    return parts >= _MIN_MACOS


def _parse_3d_translation(pt_obj: Any) -> tuple[float, float, float] | None:
    """Extract (tx, ty, tz) from a VNHumanBodyRecognizedPoint3D's 4×4 matrix repr.

    The repr format is: '4x4:|c00 c01 ... c10 c11 ... tx ty tz 1| ...'
    Columns are laid out sequentially; the last column (index 12-14) is the
    translation vector in body space (metres, y-UP convention).
    """
    try:
        s = str(pt_obj)
        nums = _MATRIX_RE.findall(s)
        # Skip leading "4" "4" from "4x4:", then read 16 matrix floats
        vals = [float(x) for x in nums[2:18]]
        if len(vals) < 15:
            return None
        # vals[12], vals[13], vals[14] = tx, ty, tz
        return vals[12], vals[13], vals[14]
    except Exception:
        return None


def _frame_to_cgimage(frame_bgr: np.ndarray) -> Any:
    """Convert BGR numpy array → CGImageRef for Vision."""
    import Foundation, Quartz
    h, w = frame_bgr.shape[:2]
    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    data = Foundation.NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    return Quartz.CGImageCreate(
        w, h, 8, 32, w * 4,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaLast,
        provider, None, False, Quartz.kCGRenderingIntentDefault,
    )


class VisionPoseExtractor:
    """Apple Vision 3D pose extractor (spike).

    Drop-in replacement for PoseExtractor:
      - Same YOLO-based person detector and crop strategy.
      - Replaces MediaPipe inference with VNDetectHumanBodyPoseRequest (2D)
        + VNDetectHumanBodyPose3DRequest (3D) on the person crop.
      - Returns FramePose with the same 33-slot landmark layout.

    Requires macOS 14+ and PyObjC (pyobjc-framework-Vision).
    """

    def __init__(self, min_visibility: float = 0.3) -> None:
        self._min_visibility = min_visibility
        self._detector = PersonDetector()
        self._pose_tracker = PersonTracker()
        self._last_timestamp_s: float | None = None
        self._yolo_ok = True
        self._vision_ok = True
        self._Vision: Any = None
        self._Foundation: Any = None
        self._Quartz: Any = None
        # Reused per-session request objects — creating new ones per frame
        # costs ~100ms extra due to ObjC allocation overhead
        self._req_2d: Any = None
        self._req_3d: Any = None

    # ------------------------------------------------------------------
    # Context manager (mirrors PoseExtractor interface)
    # ------------------------------------------------------------------

    def __enter__(self) -> "VisionPoseExtractor":
        if not _check_macos_version():
            raise RuntimeError(
                "VisionPoseExtractor requires macOS 14.0+. "
                "Current system does not meet this requirement."
            )
        try:
            import Vision, Foundation, Quartz
            self._Vision = Vision
            self._Foundation = Foundation
            self._Quartz = Quartz
        except ImportError as e:
            raise ImportError(
                "PyObjC Vision framework required. "
                "Install: pip install pyobjc-framework-Vision pyobjc-core"
            ) from e

        # Warm-up: load YOLO detector
        try:
            self._detector._ensure_loaded()
            print("[vision] Detector: YOLOv8n (YOLO crop + Apple Vision 3D)")
        except Exception as e:
            print(f"[vision] YOLO unavailable ({e}), using full-frame Vision.")
            self._yolo_ok = False

        # Create reusable request objects once — avoids ~100ms/frame ObjC alloc overhead
        try:
            self._req_2d = self._Vision.VNDetectHumanBodyPoseRequest.new()
            self._req_3d = self._Vision.VNDetectHumanBodyPose3DRequest.new()
            print("[vision] Apple Vision 3D pose ready (requests pre-allocated)")
        except Exception as e:
            print(f"[vision] Vision 3D unavailable: {e}")
            self._vision_ok = False

        return self

    def __exit__(self, *_: Any) -> None:
        pass   # Vision has no explicit cleanup

    def update_tracking(self, frame_bgr: np.ndarray) -> None:
        """YOLO-only pass for ByteTrack continuity (same interface as PoseExtractor)."""
        if self._yolo_ok:
            try:
                self._detector.detect_primary(frame_bgr)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract(
        self, frame_bgr: np.ndarray, frame_idx: int, timestamp_s: float
    ) -> FramePose | None:
        """Extract pose via Apple Vision. Returns FramePose or None."""
        if not self._vision_ok:
            return None

        dt = 0.05
        if self._last_timestamp_s is not None:
            dt = max(1e-3, timestamp_s - self._last_timestamp_s)
        self._last_timestamp_s = timestamp_s

        h, w = frame_bgr.shape[:2]
        crop_frame = frame_bgr
        crop_region: tuple[int, int, int, int] | None = None

        # YOLO crop (same strategy as PoseExtractor)
        if self._yolo_ok:
            try:
                best_bbox = self._detector.detect_primary(frame_bgr)
                if best_bbox is not None:
                    bx1, by1, bx2, by2, _ = best_bbox
                    raw_area = (bx2 - bx1) * (by2 - by1)
                    if raw_area >= 3500:
                        crop_frame, (cx1, cy1, cx2, cy2) = self._detector.crop(
                            frame_bgr, best_bbox
                        )
                        if crop_frame.size > 0:
                            crop_region = (cx1, cy1, cx2, cy2)
                        else:
                            crop_frame = frame_bgr
                else:
                    return None   # no person found → gap-filling handles it
            except Exception as e:
                print(f"[vision] YOLO error frame {frame_idx}: {e}")

        # Convert crop to CGImage
        ch, cw = crop_frame.shape[:2]
        try:
            cg_img = _frame_to_cgimage(crop_frame)
        except Exception as e:
            print(f"[vision] CGImage error frame {frame_idx}: {e}")
            return None

        # Run both requests using pre-allocated request objects
        handler = self._Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_img, {})
        ok, err = handler.performRequests_error_([self._req_2d, self._req_3d], None)
        if not ok or err:
            return None

        res_2d = self._req_2d.results()
        res_3d = self._req_3d.results()
        if not res_2d:
            return None

        obs_2d = res_2d[0]
        obs_3d = res_3d[0] if res_3d else None

        # Build sparse 33-slot landmark lists
        landmarks = [PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0)] * _N_LANDMARKS
        world_landmarks = [PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0)] * _N_LANDMARKS

        V = self._Vision

        # --- 2D landmarks (normalised crop coords → full-frame coords) ---
        for const_name, mp_idx in _JOINT_2D_TO_IDX.items():
            key = getattr(V, const_name, None)
            if key is None:
                continue
            try:
                res = obs_2d.recognizedPointForJointName_error_(key, None)
                pt = res[0] if isinstance(res, tuple) else res
                conf = float(pt.confidence())
                # Vision 2D origin is bottom-left → flip y to top-left
                vx = float(pt.x())
                vy = 1.0 - float(pt.y())

                # If we used a crop, map back to full-frame normalised coords
                if crop_region is not None:
                    cx1, cy1, cx2, cy2 = crop_region
                    vx = (cx1 + vx * cw) / w
                    vy = (cy1 + vy * ch) / h

                landmarks[mp_idx] = PoseLandmark(
                    x=vx, y=vy, z=0.0, visibility=conf
                )
            except Exception:
                pass

        # --- 3D world landmarks (metres, y-negated → y-DOWN to match MediaPipe) ---
        if obs_3d is not None:
            for const_name, mp_idx in _JOINT_3D_TO_IDX.items():
                key = getattr(V, const_name, None)
                if key is None:
                    continue
                try:
                    res = obs_3d.recognizedPointForJointName_error_(key, None)
                    pt3 = res[0] if isinstance(res, tuple) else res
                    xyz = _parse_3d_translation(pt3)
                    if xyz is not None:
                        tx, ty, tz = xyz
                        # Negate y: Vision y-UP → MediaPipe y-DOWN convention
                        # (so existing geometry.py functions work unchanged)
                        world_landmarks[mp_idx] = PoseLandmark(
                            x=tx, y=-ty, z=tz,
                            visibility=landmarks[mp_idx].visibility,
                        )
                except Exception:
                    pass

        # Confidence = mean of key joints
        key_vis = [
            landmarks[i].visibility
            for i in _CONFIDENCE_JOINTS
            if landmarks[i].visibility > 0
        ]
        pose_conf = float(np.mean(key_vis)) if key_vis else 0.0

        return FramePose(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            landmarks=landmarks,
            pose_confidence=pose_conf,
            is_smoothed=False,
            world_landmarks=world_landmarks,
            tracking_quality=1.0,
            detection_bbox=(bx1, by1, bx2, by2) if crop_region else None,
        )
