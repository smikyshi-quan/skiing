# Codex Prompt for Track C — Geometry / Scale Stabilization

Paste everything below into the Codex thread that has access to `tracks/C_geometry_scale/`.

---

Fix the 2D-to-3D coordinate transform in our alpine ski racing video analysis pipeline. This is the single biggest source of physics validation failures.

## The problem

The pipeline converts pixel coordinates to real-world meters using detected gate positions as reference points. Currently this is catastrophically wrong:
- Auto-calibration correction factors: 6x to 243x (should be close to 1.0x)
- Max speeds after transform: 419–3123 km/h (should be <70 km/h for slalom)
- Max G-forces: 355–1479 G (should be <5 G)

The auto-calibration rescales everything as a band-aid, but the underlying pixel→meter conversion is broken by 1–2 orders of magnitude.

## Root causes to investigate

1. **Gate spacing assumption** — hardcoded at `gate_spacing_m=12.0` in the pipeline constructor. FIS slalom spacing is typically 9–13m. If these are training course videos, spacing could be different. Test sensitivity by running the pipeline with gate_spacing = 8, 10, 12, 14 on each regression video. Record the auto-cal correction factor and P90 speed for each.

2. **Camera pitch correction** — hardcoded at 6 degrees (`camera_pitch_deg=6`). At 6 degrees, cos(6°) = 0.9945 which is negligible. Broadcast cameras often shoot from 20–40 degrees above the slope. Test pitch = 0, 10, 20, 30 degrees on each regression video. Find the pitch that minimizes the correction factor.

3. **Projection mode** — currently defaults to `projection="scale"` (linear Y-mapping). The alternative is full homography. Compare both on all 3 regression videos.

4. **Dynamic per-frame scale (Phase 4)** — `DynamicScaleTransform` recalculates scale each frame. Without smoothing, single-frame gate jitter creates scale spikes. Add a median filter (window=15 frames) and reject any per-frame scale >2x or <0.5x the overall median.

5. **Auto-calibration guard rails** — currently no limit. If correction > 5x, something upstream is fundamentally broken. Cap it at 5x and print a diagnostic warning listing likely causes.

## What to do

1. **Run the sensitivity sweep** — gate_spacing × camera_pitch × projection mode. Save results to `tracks/C_geometry_scale/reports/geometry_sensitivity_YYYYMMDD.md`. This is the most important first step because it tells you which parameter is most impactful.

2. **Update `ski_racing/transform.py`:**
   - Add scale smoothing (median filter, window=15) in `DynamicScaleTransform`
   - Add scale bounds (reject >2x or <0.5x median, interpolate rejected)
   - Add pitch estimation function: estimate camera pitch from ratio of gate sizes at top vs bottom of frame

3. **Update `ski_racing/pipeline.py`:**
   - In `_auto_calibrate_scale()`: cap correction at 5x. If would exceed 5x, flag video as "unable to calibrate" and print diagnostic (wrong gate spacing? wrong pitch? too few gates?)
   - Add per-frame scale time series to output JSON for debugging

4. **Update `scripts/process_video.py`:** add `--gate-spacing` and `--camera-pitch` CLI flags

## Files to read first

1. `tracks/C_geometry_scale/Track_C_Geometry_Scale_Stabilization.docx` — full spec with acceptance criteria
2. `ski_racing/transform.py` — YOUR MAIN FILE, read all of it (HomographyTransform, CameraMotionCompensator, DynamicScaleTransform)
3. `ski_racing/pipeline.py` — search for `_auto_calibrate_scale`, understand Phase 4
4. `tracks/C_geometry_scale/reports/video_regression_20260214.md` — current baseline numbers

## Files you own

- `ski_racing/transform.py` (main work)
- `ski_racing/pipeline.py` (`_auto_calibrate_scale` section)
- `scripts/process_video.py` (add CLI flags)

## Do NOT modify

- `ski_racing/tracking.py` (owned by someone else working on Track D)
- `ski_racing/detection.py` (shared, read-only)

## How to run

```bash
python scripts/process_video.py tracks/E_evaluation_ci/regression_videos/<VIDEO>.mp4 \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --summary
```

Check the output JSON for `auto_calibration.correction_factor`.

## Pass criteria

On the 3 regression videos (2907, 2909, 2911 in `tracks/E_evaluation_ci/regression_videos/`):
- Auto-cal correction factor mean < 5x (currently 84.8x)
- Auto-cal correction factor max < 10x (currently 243x)
- P90 speed before auto-cal in the 15–70 km/h range for slalom

## Note

Someone else is working on tracking/smoothing (Track D). They may also edit `pipeline.py` to change pipeline phase ordering. If you make structural changes to `pipeline.py`, document what you changed in `tracks/C_geometry_scale/reports/` so they can see it.
