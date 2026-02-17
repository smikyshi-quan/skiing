# Runtime Performance Profile - 20260215

## Scope
- Step 1 profiling only (no optimization work yet).
- Added `time.time()` instrumentation in `ski_racing/pipeline.py` around `process_video()` phases:
  - Gate detection (initial frame search)
  - Gate tracking (full video pass)
  - Perspective transform calculation
  - Camera motion compensation
  - Skier tracking (full video pass)
  - Kalman smoothing
  - 3D trajectory transform
  - Physics validation

## Regression Runs
Command used for each video:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/process_video.py \
  "tracks/E_evaluation_ci/regression_videos/<VIDEO>.mp4" \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --summary \
  --output-dir tracks/F_runtime_parallel/reports/perf_profile_20260215
```

Videos profiled:
- `2907_1765738705(Video in Original Quality).mp4`
- `2909_1765738725(Video in Original Quality).mp4`
- `2911_1765738746(Video in Original Quality).mp4`

## Per-phase Timing (seconds)

| Video | Gate detection | Gate tracking | Perspective transform | Camera motion comp | Skier tracking | Kalman smoothing | 3D transform | Physics validation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2907 | 0.088 | 35.868 | 0.000 | 0.012 | 35.529 | 0.017 | 0.011 | 0.001 |
| 2909 | 0.059 | 38.025 | 0.000 | 0.010 | 38.634 | 0.019 | 0.014 | 0.002 |
| 2911 | 0.065 | 33.742 | 0.000 | 0.009 | 33.624 | 0.015 | 0.012 | 0.001 |
| Mean | 0.071 | 35.879 | 0.000 | 0.010 | 35.929 | 0.017 | 0.012 | 0.002 |

## Total Profiled Runtime (sum of phases above)

| Video | Total profiled seconds |
|---|---:|
| 2907 | 71.528 |
| 2909 | 76.763 |
| 2911 | 67.469 |
| Mean | 71.920 |

## Bottleneck Summary (baseline)
1. Gate tracking (full video pass): mean 35.879s.
2. Skier tracking (full video pass): mean 35.929s.
3. Everything else combined: mean 0.113s.

Gate tracking + skier tracking account for ~99.8% of the profiled runtime baseline.
