"""YOLOv8-based person detector with ByteTrack + multi-signal track scoring.

Pipeline:
  1. YOLO detects all persons each frame.
  2. ByteTrack assigns consistent track IDs across frames.
  3. During warm-up, every track is scored on four normalised signals:
       centrality  — anisotropic Gaussian prior centred at (0.5, 0.6)
       motion      — EMA-smoothed per-frame displacement (active skier > static coach)
       size        — sqrt(area fraction) so close skiers don't dominate early
       age         — hit-streak, ramps to 1.0 at _MIN_AGE_TO_COMMIT frames
  4. We commit to the top-scoring track once it has led the runner-up by
     _COMMIT_MARGIN for _COMMIT_STREAK consecutive frames AND has age ≥
     _MIN_AGE_TO_COMMIT.  (mirrors SORT min_hits / DeepSORT n_init logic)
  5. After commit we follow the locked ID; if it disappears for longer than
     _RELOCK_AFTER_LOST frames we drop the lock and restart warm-up.
"""

from __future__ import annotations

import math

import numpy as np

_PERSON_CLASS = 0
_DEFAULT_MODEL = "yolov8n.pt"

# ---------------------------------------------------------------------------
# Warm-up / commit thresholds
# ---------------------------------------------------------------------------
_MIN_AGE_TO_COMMIT = 10   # track must have been seen this many frames first
_COMMIT_MARGIN     = 0.20  # top-track score must exceed runner-up by this much
_COMMIT_STREAK     = 5    # consecutive frames above margin required to commit

# Frames to hold last bbox after committed track disappears before re-lock
_RELOCK_AFTER_LOST = 10

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
_W_CENTRALITY = 0.40
_W_MOTION     = 0.35
_W_SIZE       = 0.10
_W_AGE        = 0.15

# ---------------------------------------------------------------------------
# Centrality prior  (normalised image coords, origin = top-left)
# ---------------------------------------------------------------------------
# x: centred at 0.5 — horizontal centre; wider σ because slalom zigzags
# y: biased to 0.60 — subjects appear slightly below centre in follow shots
_CENTER_X = 0.50
_CENTER_Y = 0.60
_SIGMA_X  = 0.30   # wide — handles horizontal slalom oscillation
_SIGMA_Y  = 0.20   # tighter — subject is usually in mid-lower half

# ---------------------------------------------------------------------------
# Motion (per-frame normalised-coord displacement)
# ---------------------------------------------------------------------------
# Active skier: ~0.01–0.05 / frame at 20-fps analysis.
# Static coach / spectator: ~0.001–0.005 / frame.
_V_CAP = 0.05     # displacement at which score_m saturates to 1.0
_V_EMA  = 0.40    # weight on new sample in velocity EMA  (0.4 = responsive)

# ---------------------------------------------------------------------------
# Size  (sqrt of bbox area fraction of frame area)
# ---------------------------------------------------------------------------
# Compresses the range so a nearby skier filling 12 % of frame
# scores ~1.0 rather than dominating every other signal.
_SQRT_AREA_CAP = 0.35   # sqrt(0.12) ≈ 0.35

# ---------------------------------------------------------------------------
# Score EMA across frames
# ---------------------------------------------------------------------------
_SCORE_EMA = 0.50   # weight on historical score; 0.5 = balanced smoothing

# ---------------------------------------------------------------------------
# Stale-track cleanup
# ---------------------------------------------------------------------------
_STALE_CLEANUP_AFTER = 20   # purge tracks not seen for this many frames


def _pad_bbox(
    x1: int, y1: int, x2: int, y2: int,
    pad_frac: float,
    frame_w: int, frame_h: int,
) -> tuple[int, int, int, int]:
    """Expand bbox by pad_frac of its own size on each side, clamped to frame."""
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(frame_w, x2 + px),
        min(frame_h, y2 + py),
    )


class PersonDetector:
    """YOLOv8 person detector with ByteTrack + multi-signal track scoring.

    Usage:
        for frame in video:
            bbox = detector.detect_primary(frame)   # (x1,y1,x2,y2,conf) or None
            if bbox:
                crop, region = detector.crop(frame, bbox)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        conf: float = 0.25,
        pad_frac: float = 0.20,
    ) -> None:
        self._model_name = model_name
        self._conf = conf
        self._pad_frac = pad_frac
        self._model = None

        # --- Committed-phase state ---
        self._committed_id: int | None = None
        self._committed_lost: int = 0       # frames since committed track disappeared
        self._last_bbox: tuple[int, int, int, int, float] | None = None

        # --- Per-track warm-up state (keyed by ByteTrack ID) ---
        self._track_ages:        dict[int, int]               = {}
        self._track_scores:      dict[int, float]             = {}
        self._track_prev_center: dict[int, tuple[float, float]] = {}
        self._track_smoothed_vel: dict[int, float]            = {}
        self._track_last_seen:   dict[int, int]               = {}

        # Global frame counter (incremented on every detect_primary call)
        self._frame_count: int = 0

        # Consecutive frames for which the top track has led by _COMMIT_MARGIN
        self._commit_streak: int = 0

        # When True, the next detect_primary call uses persist=False to break
        # ByteTrack's association chain across a scene cut.
        self._reset_bytetrack_pending: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_name)

    # ------------------------------------------------------------------
    # Internal: per-track score
    # ------------------------------------------------------------------

    def _score_track(
        self,
        x1: int, y1: int, x2: int, y2: int,
        tid: int,
        frame_w: int, frame_h: int,
    ) -> float:
        """Compute a single [0,1] score for one track in the current frame."""
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h

        # Centrality — anisotropic Gaussian
        dx = cx - _CENTER_X
        dy = cy - _CENTER_Y
        score_c = math.exp(
            -(dx * dx / (2 * _SIGMA_X * _SIGMA_X)
              + dy * dy / (2 * _SIGMA_Y * _SIGMA_Y))
        )

        # Motion — EMA velocity in normalised coords per frame
        vel = self._track_smoothed_vel.get(tid, 0.0)
        score_m = min(1.0, vel / _V_CAP)

        # Size — sqrt(area fraction), compressed
        area_frac = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
        score_s = min(1.0, math.sqrt(area_frac) / _SQRT_AREA_CAP)

        # Age / hit-streak
        age = self._track_ages.get(tid, 0)
        score_age = min(1.0, age / max(1, _MIN_AGE_TO_COMMIT))

        return (
            _W_CENTRALITY * score_c
            + _W_MOTION   * score_m
            + _W_SIZE     * score_s
            + _W_AGE      * score_age
        )

    # ------------------------------------------------------------------
    # Internal: state helpers
    # ------------------------------------------------------------------

    def _update_track_state(
        self,
        x1: int, y1: int, x2: int, y2: int,
        tid: int,
        frame_w: int, frame_h: int,
    ) -> None:
        """Update age, position, velocity, and last-seen for *tid*."""
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h

        prev_cx, prev_cy = self._track_prev_center.get(tid, (cx, cy))
        disp = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

        self._track_ages[tid]        = self._track_ages.get(tid, 0) + 1
        self._track_last_seen[tid]   = self._frame_count
        self._track_prev_center[tid] = (cx, cy)
        self._track_smoothed_vel[tid] = (
            (1.0 - _V_EMA) * self._track_smoothed_vel.get(tid, 0.0)
            + _V_EMA * disp
        )

    def _update_score(
        self,
        x1: int, y1: int, x2: int, y2: int,
        tid: int,
        frame_w: int, frame_h: int,
    ) -> float:
        """Compute raw score, apply EMA, store, and return smoothed score."""
        raw = self._score_track(x1, y1, x2, y2, tid, frame_w, frame_h)
        smoothed = (
            _SCORE_EMA * self._track_scores.get(tid, raw)
            + (1.0 - _SCORE_EMA) * raw
        )
        self._track_scores[tid] = smoothed
        return smoothed

    def _purge_stale_tracks(self) -> None:
        stale = [
            tid for tid, last in self._track_last_seen.items()
            if self._frame_count - last > _STALE_CLEANUP_AFTER
        ]
        for tid in stale:
            for store in (
                self._track_ages, self._track_scores,
                self._track_prev_center, self._track_smoothed_vel,
                self._track_last_seen,
            ):
                store.pop(tid, None)

    def _reset_to_warmup(self) -> None:
        """Drop the committed lock and restart warm-up scoring.

        Track history is preserved — accumulated ages and velocities give
        the scorer a head-start on re-selection.
        """
        self._committed_id   = None
        self._committed_lost = 0
        self._commit_streak  = 0

    def reset_bytetrack(self) -> None:
        """Hard reset after a scene cut.

        On the next detect_primary call, ByteTrack is re-initialised by
        running YOLO with persist=False for one frame, breaking the track-ID
        association chain from the previous shot.  Warm-up scoring state is
        also cleared so stale velocities / scores from the old shot don't
        bias selection in the new one.
        """
        self._reset_bytetrack_pending = True
        # Also clear all per-track state — old track history is meaningless
        # across a hard cut.
        self._track_ages.clear()
        self._track_scores.clear()
        self._track_prev_center.clear()
        self._track_smoothed_vel.clear()
        self._track_last_seen.clear()
        self._reset_to_warmup()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def detect_primary(
        self, frame_bgr: np.ndarray
    ) -> tuple[int, int, int, int, float] | None:
        """Return (x1,y1,x2,y2,conf) for the primary skier, or None."""
        self._ensure_loaded()
        self._frame_count += 1

        # persist=False for one frame after a scene cut — breaks ByteTrack's
        # association chain so old track IDs aren't matched to new-shot people.
        persist = True
        if self._reset_bytetrack_pending:
            persist = False
            self._reset_bytetrack_pending = False

        results = self._model.track(
            frame_bgr,
            persist=persist,
            conf=self._conf,
            classes=[_PERSON_CLASS],
            verbose=False,
        )

        frame_h, frame_w = frame_bgr.shape[:2]

        # Parse ByteTrack output
        detections: list[tuple[int, int, int, int, float, int | None]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                tid  = int(box.id[0]) if box.id is not None else None
                detections.append((x1, y1, x2, y2, conf, tid))

        # ----------------------------------------------------------
        # Committed phase — just follow the locked track
        # ----------------------------------------------------------
        if self._committed_id is not None:
            locked = [d for d in detections if d[5] == self._committed_id]
            if locked:
                self._committed_lost = 0
                self._last_bbox = locked[0][:5]
                return self._last_bbox
            # Track temporarily missing
            self._committed_lost += 1
            if self._committed_lost <= _RELOCK_AFTER_LOST:
                return self._last_bbox   # hold last known position
            # Lost too long — drop lock, restart warm-up
            self._reset_to_warmup()

        # ----------------------------------------------------------
        # No detections at all
        # ----------------------------------------------------------
        if not detections:
            return None

        # ----------------------------------------------------------
        # Warm-up phase — update state and score every track
        # ----------------------------------------------------------
        tracked: list[tuple[tuple, float]] = []   # (detection, score)

        for x1, y1, x2, y2, conf, tid in detections:
            if tid is None:
                continue
            self._update_track_state(x1, y1, x2, y2, tid, frame_w, frame_h)
            score = self._update_score(x1, y1, x2, y2, tid, frame_w, frame_h)
            tracked.append(((x1, y1, x2, y2, conf, tid), score))

        self._purge_stale_tracks()

        # Fall back to largest if ByteTrack has not assigned IDs yet
        if not tracked:
            best = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            self._last_bbox = best[:5]
            return self._last_bbox

        tracked.sort(key=lambda x: x[1], reverse=True)
        top_det,   top_score    = tracked[0]
        top_tid    = top_det[5]
        second_score = tracked[1][1] if len(tracked) >= 2 else 0.0

        # ----------------------------------------------------------
        # Dynamic commit check
        # ----------------------------------------------------------
        top_age = self._track_ages.get(top_tid, 0)
        if top_age >= _MIN_AGE_TO_COMMIT:
            margin = top_score - second_score
            if margin >= _COMMIT_MARGIN:
                self._commit_streak += 1
            else:
                self._commit_streak = 0

            if self._commit_streak >= _COMMIT_STREAK:
                self._committed_id   = top_tid
                self._committed_lost = 0
                print(
                    f"[tracker] Committed to track {top_tid}  "
                    f"score={top_score:.2f}  margin={margin:.2f}  age={top_age}"
                )

        # Return best candidate even during warm-up
        self._last_bbox = top_det[:5]
        return self._last_bbox

    # ------------------------------------------------------------------
    # Legacy detection (fallback / tests)
    # ------------------------------------------------------------------

    def detect(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[int, int, int, int, float]]:
        """Return raw list of (x1,y1,x2,y2,conf) — no tracking."""
        self._ensure_loaded()
        results = self._model(
            frame_bgr, conf=self._conf, classes=[_PERSON_CLASS], verbose=False,
        )
        boxes: list[tuple[int, int, int, int, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes

    # ------------------------------------------------------------------
    # Crop helper
    # ------------------------------------------------------------------

    def crop(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int, float],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop frame to bbox + padding."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2, _ = bbox
        cx1, cy1, cx2, cy2 = _pad_bbox(x1, y1, x2, y2, self._pad_frac, w, h)
        return frame_bgr[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)
