# Codex Prompt for Track D — Tracking / Outlier Handling

Paste everything below into the Codex thread that has access to `tracks/D_tracking_outlier/`.

---

Improve skier tracking robustness and trajectory smoothness in our alpine ski racing video analysis pipeline. Your changes happen in the 2D tracking stage, before the trajectory gets projected to 3D.

## The problem

The tracker sometimes loses the skier and latches onto spectators for a frame, producing one-frame spikes. These spikes propagate through Kalman smoothing and 3D projection, causing:
- Max position jumps: 4.78–28.94 meters per frame (should be <1m)
- Max G-forces: 355–1479 G (should be <5 G)
- ByteTrack coverage sometimes drops below 40%, triggering a weaker fallback tracker

## Current tracking architecture (already implemented — you're improving it)

Read `ski_racing/tracking.py` carefully. It contains:
- `SkierTracker`: YOLOv8n person detection + ByteTrack for persistent ID. Fallback to temporal consistency if ByteTrack coverage < 40%.
- `KalmanSmoother`: RTS backward-forward smoother. State = [x, y, vx, vy]. Process noise Q is discipline-tuned (Slalom=800, GS=400, DH=250). Innovation gating at Mahalanobis 9.0. Over-smoothing warning if turns compressed >50%.

## What to do

### 1. Add pre-smoothing outlier filter (NEW code)
Before the Kalman smoother receives the raw trajectory, filter outliers:
- Sliding-window median, window=5 frames
- For each point, compute median (x,y) in window and median absolute deviation (MAD)
- If point is >3× MAD from window median, flag as outlier
- Replace outliers with linear interpolation between nearest non-outlier neighbors
- Keep original detections in a separate debug field
- Add `outlier_count` and `outlier_frames` to the pipeline output JSON
- If outlier_count > 5% of total frames, print warning

### 2. Tune ByteTrack parameters
Current config uses ultralytics defaults. Test these settings:
- `track_high_thresh=0.3` (lower from default 0.5)
- `track_low_thresh=0.05`
- `new_track_thresh=0.4`
- `track_buffer=60` (raise from default 30)

Add re-identification after gap: if tracker loses skier for >15 frames then reacquires a person, verify it's the same person by checking:
- New detection within 200px of predicted position (extrapolate from last velocity)
- Bounding box size within 50% of last known size

Add diagnostic logging to output JSON: `bytetrack_coverage` (fraction of frames with valid track), `track_id_switches` (count of ID changes).

### 3. Kalman Q sweep
Test process noise Q = [200, 400, 800, 1600, 3200] on each regression video. For each Q, record:
- RMS difference between smoothed and raw trajectory
- Max speed after 3D projection
- Max G-force after 3D projection
Save results to `tracks/D_tracking_outlier/reports/kalman_tuning_YYYYMMDD.md`.

### 4. Consider pipeline reordering (investigate, document findings)
Current order: track → smooth → transform (which includes camera compensation)
Alternative: track → camera compensate 2D → smooth → transform
The idea: subtracting camera motion from raw 2D positions before smoothing may reduce apparent jumps. Test both orderings on one video, compare max jump and smoothness. Document your finding in `tracks/D_tracking_outlier/reports/`. Note: this changes `pipeline.py` which someone else also edits for geometry fixes (Track C), so just document your recommendation — don't force the change without noting it.

## Files to read first

1. `tracks/D_tracking_outlier/Track_D_Tracking_Outlier_Handling.docx` — full spec
2. `ski_racing/tracking.py` — YOUR MAIN FILE, read all of it
3. `ski_racing/pipeline.py` — understand phase ordering
4. `tracks/D_tracking_outlier/reports/video_regression_20260214.md` — current numbers

## Files you own

- `ski_racing/tracking.py` (main work)
- `ski_racing/visualize.py` (improve trajectory overlay for debugging)

## Files you may edit with care

- `ski_racing/pipeline.py` (only if adding diagnostic fields to output JSON or changing ordering)

## Do NOT modify

- `ski_racing/transform.py` (owned by Track C)
- `ski_racing/detection.py` (shared, read-only)

## How to run

```bash
python scripts/process_video.py tracks/E_evaluation_ci/regression_videos/<VIDEO>.mp4 \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --demo-video --summary
```

The `--demo-video` flag generates an overlay video — visually check that the trajectory trail follows the skier continuously with no jumps to spectators.

## Pass criteria (measured on 2D trajectory in pixels, NOT 3D meters)

On the 3 regression videos (2907, 2909, 2911 in `tracks/E_evaluation_ci/regression_videos/`):
- Max frame-to-frame jump: < 50 pixels
- ByteTrack coverage: > 80%
- Track-ID switches: 0
- Outlier frames: < 5% of total
