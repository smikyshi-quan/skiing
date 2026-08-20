"""Two-step pose extractor: YOLOv8 person detection → MediaPipe on crop."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from technique_analysis.common.contracts.models import FramePose, PoseLandmark
from technique_analysis.common.pose.person_detector import PersonDetector
from technique_analysis.common.pose.tracker import PersonTracker
from technique_analysis.common.pose import rotation_recovery

# Key joint indices for confidence scoring
_CONFIDENCE_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28]

# COCO-17 keypoint index -> MediaPipe-33 landmark index.
_COCO_TO_MEDIAPIPE = {
    0: 0,    # nose
    1: 2,    # left eye -> MediaPipe left eye
    2: 5,    # right eye -> MediaPipe right eye
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}


# ---------------------------------------------------------------------------
# Scene cut detector
# ---------------------------------------------------------------------------

class _SceneCutDetector:
    """Cheap scene cut detector using mean-absolute-difference on a small
    grayscale thumbnail with a rolling robust threshold.

    Mirrors the PySceneDetect AdaptiveDetector pattern:
      - Compute MAD between consecutive 64×36 grayscale frames.
      - Maintain a rolling buffer of recent MAD values.
      - Fire a cut when the current MAD exceeds  median + K * spread
        (spread = median absolute deviation of the buffer — robust to outliers).
      - A short cooldown prevents re-triggering immediately after a cut.
    """

    _SMALL_W  = 64
    _SMALL_H  = 36
    _BUFFER   = 30     # rolling window length for robust threshold
    _K_SIGMA  = 3.0    # threshold multiplier on the robust spread
    _MIN_MAD  = 12.0   # absolute floor — ignores nearly-static scenes
    _COOLDOWN = 15     # frames suppressed after a confirmed cut

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._mad_buffer: collections.deque = collections.deque(
            maxlen=self._BUFFER
        )
        self._cooldown_remaining: int = 0

    def is_cut(self, frame_bgr: np.ndarray) -> bool:
        """Return True if *frame_bgr* looks like the first frame of a new shot."""
        small = cv2.resize(
            frame_bgr, (self._SMALL_W, self._SMALL_H),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        mad = float(np.mean(np.abs(gray - self._prev_gray)))
        self._prev_gray = gray
        self._mad_buffer.append(mad)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        if len(self._mad_buffer) < 5:
            return False   # not enough history yet

        buf    = np.array(self._mad_buffer)
        median = float(np.median(buf))
        spread = float(np.median(np.abs(buf - median)))
        threshold = median + self._K_SIGMA * max(spread, 1.0)

        if mad > threshold and mad > self._MIN_MAD:
            self._cooldown_remaining = self._COOLDOWN
            return True

        return False

_MODEL_PREFERENCE = ["pose_landmarker_full.task", "pose_landmarker_lite.task"]
_FULL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_YOLO_POSE_MODEL = "yolov8n-pose.pt"


def _find_or_download_model() -> Path:
    """Locate best available pose model; auto-download full if missing."""
    model_dir = Path(__file__).parent
    full_local = model_dir / "pose_landmarker_full.task"
    if full_local.exists():
        return full_local
    try:
        import mediapipe as mp
        pkg_dir = Path(mp.__file__).parent
        for name in _MODEL_PREFERENCE:
            candidates = list(pkg_dir.rglob(name))
            if candidates:
                return candidates[0]
    except Exception:
        pass
    try:
        import urllib.request
        print("[pose] Downloading pose_landmarker_full.task…")
        urllib.request.urlretrieve(_FULL_MODEL_URL, full_local)
        print(f"[pose]   Saved to {full_local}")
        return full_local
    except Exception as e:
        print(f"[pose]   Download failed ({e}), falling back to lite model.")
    lite_local = model_dir / "pose_landmarker_lite.task"
    if lite_local.exists():
        return lite_local
    raise FileNotFoundError(
        "No pose model found. Download pose_landmarker_full.task from:\n"
        f"  {_FULL_MODEL_URL}"
    )


def _find_yolo_pose_model() -> str:
    """Return a local YOLO pose model path when present, else model name.

    Passing the bare model name lets Ultralytics download it on machines with
    network access. Local path lookup keeps worker behavior stable regardless
    of the process working directory.
    """
    repo_root = Path(__file__).resolve().parents[5]
    for candidate in (
        Path.cwd() / _YOLO_POSE_MODEL,
        repo_root / _YOLO_POSE_MODEL,
        repo_root / "MVP" / _YOLO_POSE_MODEL,
    ):
        if candidate.exists():
            return str(candidate)
    return _YOLO_POSE_MODEL


def _transform_landmarks(
    landmarks: list[PoseLandmark],
    cx1: int, cy1: int, cx2: int, cy2: int,
    frame_w: int, frame_h: int,
) -> list[PoseLandmark]:
    """Convert landmarks from crop-normalised to full-frame-normalised coords."""
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    return [
        PoseLandmark(
            x=(cx1 + lm.x * crop_w) / frame_w,
            y=(cy1 + lm.y * crop_h) / frame_h,
            z=lm.z,
            visibility=lm.visibility,
        )
        for lm in landmarks
    ]


# Minimum uncommitted gap (seconds) that signals a new athlete epoch.
# Gaps shorter than this are treated as brief re-locks of the same person.
_NEW_SEGMENT_GAP_S = 2.0


class PoseExtractor:
    """Two-step pose extractor: YOLOv8 person detector → MediaPipe on crop.

    Pipeline per frame:
      1. YOLOv8 detects all persons → bounding boxes.
      2. Kalman BBox tracker selects the primary skier's box.
      3. Frame is cropped to that box + 20 % padding.
      4. MediaPipe runs on the crop (person fills the frame → higher accuracy).
      5. Landmarks are transformed back to full-frame normalised coordinates.
      6. World landmarks are returned as-is (metric space, unaffected by crop).

    Falls back to full-frame MediaPipe if YOLOv8 is unavailable or returns
    no person for the current frame.
    """

    # Minimum bbox height as a fraction of analysis frame height.
    # MediaPipe internally resizes crops to 224px (detector) / 256px (landmarker),
    # so a person shorter than ~7% of frame height produces an 8×-upscaled crop
    # with severe blur artefacts.  7% ≈ 75px at 1080p, 34px at 480p.
    _MIN_BBOX_HEIGHT_FRAC: float = 0.07
    # Absolute floor — never pass a crop shorter than this regardless of resolution.
    _MIN_BBOX_HEIGHT_PX: int = 40
    # After this many consecutive frames with no detection passing the gate,
    # temporarily halve the height threshold so a distant racer can slip through.
    # Metrics are still suppressed until pose confidence recovers.
    _ADAPTIVE_FALLBACK_AFTER: int = 30
    _ADAPTIVE_HEIGHT_FRAC: float = 0.035   # ~half of normal minimum

    def __init__(self, min_visibility: float = 0.5) -> None:
        self._min_visibility = min_visibility
        self._landmarker: Any = None
        self._yolo_pose_model: Any = None
        self._yolo_pose_ok = False
        self._yolo_pose_load_attempted = False
        self._mp: Any = None
        self._pose_unavailable_reason: str | None = None
        self._detector = PersonDetector()  # owns ByteTrack locking internally
        self._pose_tracker = PersonTracker()     # fallback: tracks hip midpoints
        self._last_timestamp_s: float | None = None
        self._yolo_ok = True   # flips to False on repeated YOLO failures
        self._frames_since_detection: int = 0  # for adaptive size-gate fallback
        self._cut_detector = _SceneCutDetector()
        self.scene_cuts_detected: int = 0      # reported in quality warnings

        # Segment boundary tracking — Phase 4
        # A new boundary is recorded when the tracker commits to a new track
        # after being uncommitted for >= _NEW_SEGMENT_GAP_S seconds.
        # segment_boundaries always starts with 0.0 (video start).
        self.segment_boundaries: list[float] = [0.0]
        self._prev_committed_id: int | None = None
        self._uncommitted_since_ts: float | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PoseExtractor":
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise ImportError(
                "mediapipe is required. Install with: pip install 'mediapipe>=0.10.14'"
            ) from e

        model_path = _find_or_download_model()
        print(f"[pose] Model: {model_path.name}")

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        # With two-step cropping we only need one pose per crop
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self._mp = mp
        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except RuntimeError as exc:
            self._pose_unavailable_reason = str(exc)
            self._landmarker = None
            print(
                "[pose] MediaPipe pose unavailable; continuing with "
                "YOLO pose fallback."
            )
            self._ensure_yolo_pose_loaded()

        # Warm up YOLOv8 (triggers model download once, if needed)
        try:
            self._detector._ensure_loaded()
            print("[pose] Detector: YOLOv8n (two-step pipeline active)")
        except Exception as e:
            print(f"[pose] YOLOv8 unavailable ({e}), using full-frame fallback.")
            self._yolo_ok = False

        return self

    def __exit__(self, *_: Any) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _ensure_yolo_pose_loaded(self) -> None:
        """Load a portable YOLO pose fallback when MediaPipe cannot run."""
        if self._yolo_pose_model is not None or self._yolo_pose_ok:
            return
        if self._yolo_pose_load_attempted:
            return
        self._yolo_pose_load_attempted = True
        try:
            from ultralytics import YOLO
            self._yolo_pose_model = YOLO(_find_yolo_pose_model())
            self._yolo_pose_ok = True
            print("[pose] Detector: YOLOv8n-pose fallback active")
        except Exception as exc:
            self._yolo_pose_ok = False
            self._yolo_pose_model = None
            print(f"[pose] YOLO pose fallback unavailable ({exc})")

    def update_tracking(self, frame_bgr: np.ndarray) -> None:
        """YOLO-only pass: advances ByteTrack state without running MediaPipe.

        Called on intermediate frames (between analysis frames) so ByteTrack
        sees continuous motion rather than large time jumps.
        """
        if self._cut_detector.is_cut(frame_bgr):
            self.scene_cuts_detected += 1
            self._detector.reset_bytetrack()
            print(f"[tracker] Scene cut #{self.scene_cuts_detected} detected — resetting tracker")
        if self._yolo_ok:
            try:
                self._detector.detect_primary(frame_bgr)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal: segment boundary tracking
    # ------------------------------------------------------------------

    def _update_segment_state(self, timestamp_s: float) -> None:
        """Track committed/uncommitted transitions to detect athlete switches.

        A new segment boundary is recorded when the tracker commits to a
        (possibly new) track after being uncommitted for >= _NEW_SEGMENT_GAP_S.
        Short re-locks (same athlete briefly lost) don't create new segments.
        """
        current = self._detector.committed_id

        if self._prev_committed_id is None and current is not None:
            # Just committed — was the preceding gap long enough?
            if self._uncommitted_since_ts is not None:
                gap = timestamp_s - self._uncommitted_since_ts
                if gap >= _NEW_SEGMENT_GAP_S:
                    self.segment_boundaries.append(timestamp_s)
            self._uncommitted_since_ts = None

        elif self._prev_committed_id is not None and current is None:
            # Just went uncommitted — record when it started
            if self._uncommitted_since_ts is None:
                self._uncommitted_since_ts = timestamp_s

        self._prev_committed_id = current

    # ------------------------------------------------------------------
    # Internal MediaPipe call
    # ------------------------------------------------------------------

    def _run_mediapipe(self, frame_bgr: np.ndarray) -> Any:
        """Run the MediaPipe landmarker on a BGR frame. Returns raw result."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        return self._landmarker.detect(mp_image)

    def _run_mediapipe_landmarks(
        self, frame_bgr: np.ndarray
    ) -> list[PoseLandmark] | None:
        """Run MediaPipe and return crop-normalised landmarks, or None on failure.

        Used as the inference callable passed to rotation_recovery.recover_landmarks().
        """
        try:
            result = self._run_mediapipe(frame_bgr)
        except Exception:
            return None
        if not result.pose_landmarks:
            return None
        return [
            PoseLandmark(
                x=float(lm.x), y=float(lm.y), z=float(lm.z),
                visibility=float(getattr(lm, "visibility", 1.0)),
            )
            for lm in result.pose_landmarks[0]
        ]

    def _make_frame_pose(
        self,
        landmarks: list[PoseLandmark],
        world_landmarks: list[PoseLandmark] | None,
        frame_idx: int,
        timestamp_s: float,
        detection_bbox: tuple[int, int, int, int] | None = None,
    ) -> FramePose:
        key_vis = [
            landmarks[i].visibility
            for i in _CONFIDENCE_JOINTS
            if i < len(landmarks)
        ]
        conf = float(np.mean(key_vis)) if key_vis else 0.0
        return FramePose(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            landmarks=landmarks,
            pose_confidence=conf,
            is_smoothed=False,
            world_landmarks=world_landmarks,
            tracking_quality=1.0,
            detection_bbox=detection_bbox,
        )

    def _make_pseudo_world_landmarks(
        self,
        landmarks: list[PoseLandmark],
    ) -> list[PoseLandmark] | None:
        """Approximate hip-centered world landmarks from 2D YOLO pose output.

        YOLO pose is 2D only. The rest of this pipeline expects MediaPipe-style
        world landmarks for lateral shift and turn segmentation, so this keeps
        x/y in normalized image units, centers them at the hip midpoint, and
        sets z=0. These coordinates are not metric 3D, but they allow the run
        to produce skeleton-derived metrics instead of failing outright.
        """
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        if left_hip.visibility < self._min_visibility or right_hip.visibility < self._min_visibility:
            return None
        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        return [
            PoseLandmark(
                x=lm.x - hip_x,
                y=lm.y - hip_y,
                z=0.0,
                visibility=lm.visibility,
            )
            for lm in landmarks
        ]

    def _run_yolo_pose_fallback(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        primary_bbox: tuple[int, int, int, int, float] | None,
    ) -> FramePose | None:
        """Run YOLOv8 pose and return a FramePose in the MediaPipe contract."""
        if not self._yolo_pose_ok:
            self._ensure_yolo_pose_loaded()
        if self._yolo_pose_model is None:
            return None

        try:
            results = self._yolo_pose_model(
                frame_bgr,
                conf=0.25,
                verbose=False,
                device="cpu",
            )
        except Exception as exc:
            print(f"[pose] YOLO pose error frame {frame_idx}: {exc}")
            self._yolo_pose_ok = False
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        candidates: list[tuple[float, Any, Any]] = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            for idx, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = map(float, box[:4])
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                score = area
                if primary_bbox is not None:
                    px1, py1, px2, py2, _ = primary_bbox
                    ix1, iy1 = max(x1, px1), max(y1, py1)
                    ix2, iy2 = min(x2, px2), min(y2, py2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    union = area + max(0.0, px2 - px1) * max(0.0, py2 - py1) - inter
                    score = inter / union if union > 0 else 0.0
                candidates.append((score, box, (keypoints_xy[idx], keypoints_conf[idx])))

        if not candidates:
            return None

        _, box, keypoints = max(candidates, key=lambda item: item[0])
        points_xy, points_conf = keypoints
        landmarks = [
            PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0)
            for _ in range(33)
        ]
        for coco_idx, mp_idx in _COCO_TO_MEDIAPIPE.items():
            if coco_idx >= len(points_xy):
                continue
            x_px, y_px = points_xy[coco_idx]
            conf = float(points_conf[coco_idx])
            landmarks[mp_idx] = PoseLandmark(
                x=float(x_px) / frame_w,
                y=float(y_px) / frame_h,
                z=0.0,
                visibility=conf,
            )

        key_vis = [
            landmarks[i].visibility
            for i in _CONFIDENCE_JOINTS
            if i < len(landmarks) and landmarks[i].visibility > 0
        ]
        if not key_vis:
            return None

        x1, y1, x2, y2 = map(int, box[:4])
        self._frames_since_detection = 0
        self._update_segment_state(timestamp_s)
        return self._make_frame_pose(
            landmarks,
            self._make_pseudo_world_landmarks(landmarks),
            frame_idx,
            timestamp_s,
            detection_bbox=(x1, y1, x2, y2),
        )

    # ------------------------------------------------------------------
    # Public extraction entry point
    # ------------------------------------------------------------------

    def extract(
        self, frame_bgr: np.ndarray, frame_idx: int, timestamp_s: float
    ) -> FramePose | None:
        """Extract pose from one BGR frame. Returns None if no person found."""
        if self._landmarker is None:
            if self._pose_unavailable_reason is not None:
                if self._cut_detector.is_cut(frame_bgr):
                    self.scene_cuts_detected += 1
                    self._detector.reset_bytetrack()
                    print(f"[tracker] Scene cut #{self.scene_cuts_detected} detected — resetting tracker")
                primary_bbox = None
                if self._yolo_ok:
                    try:
                        primary_bbox = self._detector.detect_primary(frame_bgr)
                    except Exception:
                        pass
                pose = self._run_yolo_pose_fallback(
                    frame_bgr, frame_idx, timestamp_s, primary_bbox
                )
                if pose is not None:
                    return pose
                self._frames_since_detection += 1
                self._update_segment_state(timestamp_s)
                return None
            raise RuntimeError("PoseExtractor must be used as a context manager.")

        dt = 0.05
        if self._last_timestamp_s is not None:
            dt = max(1e-3, timestamp_s - self._last_timestamp_s)
        self._last_timestamp_s = timestamp_s

        h, w = frame_bgr.shape[:2]

        # Scene cut check — must run before YOLO so the tracker is reset
        # before ByteTrack tries to associate across the cut boundary.
        if self._cut_detector.is_cut(frame_bgr):
            self.scene_cuts_detected += 1
            self._detector.reset_bytetrack()
            print(f"[tracker] Scene cut #{self.scene_cuts_detected} detected — resetting tracker")

        # ------ Two-step path (YOLOv8 ByteTrack → crop → MediaPipe) ---------
        if self._yolo_ok:
            try:
                best_bbox = self._detector.detect_primary(frame_bgr)
                if best_bbox is not None:
                    bx1, by1, bx2, by2, bconf = best_bbox
                    bbox_h = by2 - by1

                    # Resolution-invariant height gate.
                    # MediaPipe internally resizes crops to 256px; crops shorter
                    # than ~7% of frame height are upsampled 8× or more, producing
                    # blur artefacts that make pose estimation unreliable.
                    # After _ADAPTIVE_FALLBACK_AFTER consecutive missed frames the
                    # threshold is halved so a distant skier can slip through —
                    # metrics remain suppressed until pose confidence recovers.
                    if self._frames_since_detection >= self._ADAPTIVE_FALLBACK_AFTER:
                        min_h_frac = self._ADAPTIVE_HEIGHT_FRAC
                    else:
                        min_h_frac = self._MIN_BBOX_HEIGHT_FRAC
                    min_h = max(self._MIN_BBOX_HEIGHT_PX, int(h * min_h_frac))
                    if bbox_h < min_h:
                        self._frames_since_detection += 1
                        return None

                    crop, region = self._detector.crop(frame_bgr, best_bbox)
                    cx1, cy1, cx2, cy2 = region

                    if crop.size == 0:
                        return None

                    result = self._run_mediapipe(crop)
                    if not result.pose_landmarks:
                        return None

                    raw_lms = result.pose_landmarks[0]
                    crop_landmarks = [
                        PoseLandmark(
                            x=float(lm.x), y=float(lm.y), z=float(lm.z),
                            visibility=float(getattr(lm, "visibility", 1.0)),
                        )
                        for lm in raw_lms
                    ]

                    # Rotation recovery — triggered when primary inference is
                    # low-confidence (e.g. pole-plant moments, out-of-balance).
                    # Operates in crop-normalised space before transform.
                    key_vis = [
                        crop_landmarks[i].visibility
                        for i in _CONFIDENCE_JOINTS
                        if i < len(crop_landmarks)
                    ]
                    primary_conf = float(np.mean(key_vis)) if key_vis else 0.0
                    if primary_conf < rotation_recovery.CONF_TRIGGER:
                        crop_landmarks = rotation_recovery.recover_landmarks(
                            crop, self._run_mediapipe_landmarks, crop_landmarks
                        )

                    landmarks = _transform_landmarks(
                        crop_landmarks, cx1, cy1, cx2, cy2, w, h,
                    )

                    world_landmarks: list[PoseLandmark] | None = None
                    if result.pose_world_landmarks:
                        world_landmarks = [
                            PoseLandmark(
                                x=float(lm.x), y=float(lm.y), z=float(lm.z),
                                visibility=float(getattr(lm, "visibility", 1.0)),
                            )
                            for lm in result.pose_world_landmarks[0]
                        ]

                    self._frames_since_detection = 0
                    self._update_segment_state(timestamp_s)
                    return self._make_frame_pose(
                        landmarks, world_landmarks, frame_idx, timestamp_s,
                        detection_bbox=(bx1, by1, bx2, by2),
                    )
                # No primary person found — return None for gap-filling
                self._frames_since_detection += 1
                self._update_segment_state(timestamp_s)
                return None

            except Exception as e:
                print(f"[pose] Two-step error frame {frame_idx}: {e}. Using fallback.")
                self._yolo_ok = False

        # ------ Fallback: full-frame MediaPipe with hip tracker -----------
        result = self._run_mediapipe(frame_bgr)
        if not result.pose_landmarks:
            return None

        all_lms = list(result.pose_landmarks)
        landmark_lists = [
            [
                PoseLandmark(
                    x=float(lm.x), y=float(lm.y), z=float(lm.z),
                    visibility=float(getattr(lm, "visibility", 1.0)),
                )
                for lm in person_lms
            ]
            for person_lms in all_lms
        ]
        best_idx = self._pose_tracker.select_best(landmark_lists, dt=dt)
        landmarks = landmark_lists[best_idx]

        world_landmarks = None
        if result.pose_world_landmarks and best_idx < len(result.pose_world_landmarks):
            world_landmarks = [
                PoseLandmark(
                    x=float(lm.x), y=float(lm.y), z=float(lm.z),
                    visibility=float(getattr(lm, "visibility", 1.0)),
                )
                for lm in result.pose_world_landmarks[best_idx]
            ]

        return self._make_frame_pose(
            landmarks, world_landmarks, frame_idx, timestamp_s
        )
