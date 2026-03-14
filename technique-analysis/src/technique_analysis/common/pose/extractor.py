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

# Key joint indices for confidence scoring
_CONFIDENCE_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28]


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
        self._mp: Any = None
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

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        # With two-step cropping we only need one pose per crop
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

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

    # ------------------------------------------------------------------
    # Public extraction entry point
    # ------------------------------------------------------------------

    def extract(
        self, frame_bgr: np.ndarray, frame_idx: int, timestamp_s: float
    ) -> FramePose | None:
        """Extract pose from one BGR frame. Returns None if no person found."""
        if self._landmarker is None:
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
                    landmarks = _transform_landmarks(
                        [
                            PoseLandmark(
                                x=float(lm.x), y=float(lm.y), z=float(lm.z),
                                visibility=float(getattr(lm, "visibility", 1.0)),
                            )
                            for lm in raw_lms
                        ],
                        cx1, cy1, cx2, cy2, w, h,
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
