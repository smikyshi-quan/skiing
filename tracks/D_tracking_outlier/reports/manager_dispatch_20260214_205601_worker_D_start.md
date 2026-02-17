# Manager Dispatch: Worker D START

Timestamp: 2026-02-14 20:56:01
Priority: Phase 2 (parallel with C)
Status: ACTIVE

Use the exact prompt below.

```text
You are working on an alpine ski racing video analysis project.
Your job: make the skier tracking smoother and more robust. Reduce trajectory noise
and discontinuities in the 2D tracking stage BEFORE the trajectory hits 3D projection.
 
== THE PROBLEM ==
The tracker sometimes loses the skier and latches onto spectators, producing one-frame
spikes. These spikes propagate through Kalman smoothing and 3D projection, causing:
  - Max position jumps: 4.78-28.94 meters per frame (should be <1m)
  - Max G-forces: 355-1479 G (should be <5G)
  - ByteTrack coverage sometimes drops below 40%, triggering a weaker fallback
 
== CURRENT TRACKING ARCHITECTURE (already implemented, you're IMPROVING it) ==
  - Person detection: YOLOv8n, person class only
  - ID tracking: ByteTrack (ultralytics), fallback to temporal consistency if <40% coverage
  - Max jump filter: 25% of min(W,H) or 60% of max(W,H)
  - Kalman smoother: RTS backward-forward, discipline-tuned Q (SL=800, GS=400, DH=250)
  - 3D sanitization: remove non-finite, interpolate gaps
 
== YOUR DELIVERABLES ==
1. Pre-smoothing outlier filter (NEW):
   Sliding-window median (window=5). Flag points >3x MAD from window median.
   Replace with linear interpolation. Keep originals in debug field.
   Log outlier_count and outlier_frames to output JSON.
 
2. ByteTrack tuning:
   Test: track_high_thresh=0.3, track_low_thresh=0.05, new_track_thresh=0.4,
   track_buffer=60. Add re-ID after gap (check position within 200px of predicted,
   bbox size within 50% of last known).
   Log bytetrack_coverage and track_id_switches to output JSON.
 
3. Kalman Q sweep:
   Test Q=[200, 400, 800, 1600, 3200] on each regression video.
   For each Q: record RMS diff (smoothed vs raw), max speed, max G-force.
   Save results to tracks/D_tracking_outlier/reports/kalman_tuning_YYYYMMDD.md
 
4. Consider moving camera compensation BEFORE Kalman smoothing:
   Current: track -> smooth -> transform (includes camera comp)
   Proposed: track -> camera compensate 2D -> smooth -> transform
   Test both orderings. Note: this changes pipeline.py which Worker C also edits.
   Document your finding in tracks/D_tracking_outlier/reports/.
 
== FILES TO READ FIRST ==
1. tracks/D_tracking_outlier/Track_D_Tracking_Outlier_Handling.docx  (full spec)
2. tracks/D_tracking_outlier/README.md
3. ski_racing/tracking.py  (YOUR MAIN FILE - read ALL of it, especially
   SkierTracker, KalmanSmoother, _track_with_bytetrack_reassociate)
4. ski_racing/pipeline.py  (understand phase ordering)
5. tracks/D_tracking_outlier/reports/video_regression_20260214.md  (baseline)
 
== FILES YOU OWN ==
- ski_racing/tracking.py  (main work)
- ski_racing/visualize.py  (improve trajectory overlay for debugging)
 
== FILES YOU MAY EDIT (with care) ==
- ski_racing/pipeline.py  (only if changing pipeline ordering)
 
== DO NOT MODIFY ==
- ski_racing/transform.py  (owned by Worker C)
- ski_racing/detection.py  (shared, read-only)
 
== VISUAL CHECK ==
For each regression video, generate a demo overlay (--demo-video flag).
The trajectory trail must follow the skier continuously with no jumps.
 
== PASS CRITERIA (all measured on 2D trajectory, NOT 3D) ==
On the 3 regression videos (in tracks/E_evaluation_ci/regression_videos/):
  - Max frame-to-frame jump: < 50 pixels
  - ByteTrack coverage: > 80%
  - Track-ID switches: 0
  - Outlier frames: < 5% of total
 
== HOW TO RUN ==
python scripts/process_video.py <video_path> \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --demo-video --summary
```
