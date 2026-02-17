# Sin(Pitch) Fix Eval Report (2026-02-15)

## Code Change

Applied in `/Users/quan/Documents/personal/Stanford application project/ski_racing/transform.py`:
- `DynamicScaleTransform._compute_effective_spacing()` now always returns raw `gate_spacing_m` (no `sin(camera_pitch_deg)` scaling).
- `HomographyTransform._calculate_scale_from_gates()` now uses raw `gate_spacing_m` (no `sin(camera_pitch_deg)` scaling).
- `camera_pitch_deg` remains in constructor/CLI and is retained as metadata.

## Commands Run

### Slalom regression eval (3 videos)
```bash
python3 scripts/run_eval.py \
  --model models/gate_detector_best.pt \
  --output-root tracks/C_geometry_scale/reports
```

Artifacts:
- `/Users/quan/Documents/personal/Stanford application project/tracks/C_geometry_scale/reports/eval_20260215_1111/stage2_regression.json`
- `/Users/quan/Documents/personal/Stanford application project/tracks/C_geometry_scale/reports/eval_20260215_1111/stage2_regression/2907_analysis.json`
- `/Users/quan/Documents/personal/Stanford application project/tracks/C_geometry_scale/reports/eval_20260215_1111/stage2_regression/2909_analysis.json`
- `/Users/quan/Documents/personal/Stanford application project/tracks/C_geometry_scale/reports/eval_20260215_1111/stage2_regression/2911_analysis.json`

Note: this repo's `scripts/run_eval.py` interface uses `--model`/`--output-root` and frozen regression discovery, not `--gate-model --videos`.

### Giant slalom spot check (1 video)
```bash
python3 scripts/process_video.py \
  "tracks/E_evaluation_ci/regression_videos/2907_1765738705(Video in Original Quality).mp4" \
  --gate-model models/gate_detector_best.pt \
  --discipline giant_slalom --stabilize --summary \
  --output-dir tracks/C_geometry_scale/reports/eval_post_sinfix_20260215_gs
```

Artifact:
- `/Users/quan/Documents/personal/Stanford application project/tracks/C_geometry_scale/reports/eval_post_sinfix_20260215_gs/2907_1765738705(Video in Original Quality)_analysis.json`

## Results

### Slalom (pre-auto-cal metrics)

| Video | Pre-auto-cal P90 speed (km/h) | Auto-cal correction factor | Auto-cal applied | Status |
|---|---:|---:|---|---|
| 2907 | 2725.48 | 49.55x | No (`correction_exceeds_cap`) | FAIL |
| 2909 | 2661.14 | 48.38x | No (`correction_exceeds_cap`) | FAIL |
| 2911 | 3355.60 | 61.01x | No (`correction_exceeds_cap`) | FAIL |

### Slalom (post-transform physics summary from stage2 aggregate)

- Mean P90 speed: 1472.05 km/h
- Mean max speed: 116310.53 km/h
- Mean auto-cal correction: 52.98x
- Mean physics issue count: 5.0

### Giant slalom spot check (video 2907)

- Pre-auto-cal P90 speed: 6115.71 km/h
- Auto-cal correction factor: 76.45x (`correction_exceeds_cap`)
- Post-transform P90 speed: 2582.62 km/h
- Post-transform max speed: 470365.29 km/h

## Pass Criteria Check

1. Pre-auto-cal P90 speed 15-70 km/h on all 3 slalom videos: **FAIL**
2. Auto-cal correction factor < 2.0x: **FAIL**
3. No physics validation failures for excessive speed pre-auto-cal: **FAIL**

## Conclusion

The `sin(pitch)` spacing bug was removed in both target locations, but regression metrics remain far outside the required range. Further scale calibration work is still required (gate spacing assumptions and/or upstream gate geometry quality).
