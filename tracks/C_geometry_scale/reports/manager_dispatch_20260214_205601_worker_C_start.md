# Manager Dispatch: Worker C START

Timestamp: 2026-02-14 20:56:01
Priority: Phase 2 (parallel with D)
Status: ACTIVE

Use the exact prompt below.

```text
You are working on an alpine ski racing video analysis project.
Your job: fix the 2D-to-3D coordinate transform so that pixel positions convert to
physically plausible real-world meters without needing massive auto-calibration corrections.
 
== THE PROBLEM ==
The pipeline converts pixel coords to meters using gate positions as reference points.
Currently this is catastrophically wrong:
  - Auto-calibration correction factors: 6x to 243x (should be ~1.0x)
  - Max speeds after transform: 419-3123 km/h (should be <70 km/h for slalom)
  - Max G-forces: 355-1479 G (should be <5 G)
The auto-calibration band-aid rescales everything, but the underlying transform is broken.
 
== ROOT CAUSES TO INVESTIGATE ==
1. Gate spacing assumption: hardcoded at 12m. May be wrong for these videos.
   -> Sweep: test 8, 10, 12, 14m on all 3 regression videos.
2. Camera pitch correction: hardcoded at 6 degrees. Likely too low for broadcast angles.
   -> Sweep: test 0, 10, 20, 30 degrees on all 3 regression videos.
3. Projection mode: 'scale' (default) vs 'homography'. Compare both.
4. Dynamic per-frame scale (Phase 4): amplifies noise without smoothing.
   -> Add median filter (window=15) and bounds (reject >2x or <0.5x median).
5. Auto-cal has no guard rails. Cap at 5x and add diagnostic warnings.
 
== YOUR DELIVERABLES ==
1. Sensitivity analysis saved to tracks/C_geometry_scale/reports/
   geometry_sensitivity_YYYYMMDD.md (gate_spacing x camera_pitch x projection sweeps)
2. Updated ski_racing/transform.py: scale smoothing, scale bounds, pitch estimation
3. Updated ski_racing/pipeline.py: auto-cal capped at 5x, diagnostic warnings
4. Per-frame scale time series added to output JSON for debugging
5. Updated scripts/process_video.py: add --gate-spacing and --camera-pitch flags
 
== FILES TO READ FIRST ==
1. tracks/C_geometry_scale/Track_C_Geometry_Scale_Stabilization.docx  (full spec)
2. tracks/C_geometry_scale/README.md
3. ski_racing/transform.py  (YOUR MAIN FILE - read ALL of it)
4. ski_racing/pipeline.py  (search for _auto_calibrate_scale method)
5. tracks/C_geometry_scale/reports/video_regression_20260214.md  (current numbers)
 
== FILES YOU OWN ==
- ski_racing/transform.py  (main work)
- ski_racing/pipeline.py  (_auto_calibrate_scale section only)
- scripts/process_video.py  (add CLI flags)
 
== DO NOT MODIFY ==
- ski_racing/tracking.py  (owned by Worker D)
- ski_racing/detection.py  (shared, read-only)
 
== COORDINATE WITH WORKER D ==
If you change pipeline ordering (e.g., where camera compensation runs),
note this in tracks/C_geometry_scale/reports/ so Worker D is aware.
 
== PASS CRITERIA ==
On the 3 regression videos (in tracks/E_evaluation_ci/regression_videos/):
  - Auto-cal correction factor mean < 5x  (currently 84.8x)
  - Auto-cal correction factor max < 10x  (currently 243x)
  - P90 speed before auto-cal: 15-70 km/h for slalom
 
== HOW TO RUN ==
python scripts/process_video.py <video_path> \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --summary
```
