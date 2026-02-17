# Video Regression Comparison (New vs Previous Gate Model)

## Setup
- New model: `models/gate_detector_best.pt`
- Previous model: `models/gate_detector_best_20260213_2337_backup.pt`
- Videos: 3 unseen videos (`2907`, `2909`, `2911`)
- Pipeline settings: stabilize on, affine camera mode, scale projection, gate_conf=0.35, gate_iou=0.55

## Per-video Comparison

| Video | Gates (new/old) | Coverage (new/old) | Physics issues (new/old) | P90 speed km/h (new/old) | Max speed km/h (new/old) | Max G (new/old) | Max jump m (new/old) | Auto-cal correction (new/old) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2907_1765738705(Video in Original Quality)_analysis.json` | 7/5 | 0.969/0.969 | 5/5 | 18.5/23.3 | 3123.0/237.5 | 355.2/227.3 | 28.94/2.28 | 6.08/5.20 |
| `2909_1765738725(Video in Original Quality)_analysis.json` | 4/5 | 0.926/0.926 | 5/4 | 32.7/32.2 | 1007.4/481.8 | 1479.2/718.8 | 9.34/4.47 | 5.75/2.67 |
| `2911_1765738746(Video in Original Quality)_analysis.json` | 5/5 | 0.919/0.919 | 5/6 | 16.8/12.4 | 419.4/450.1 | 709.3/590.2 | 4.78/8.20 | 242.49/243.47 |

## Aggregate (mean across 3 videos)

| Metric | New | Old | Delta (new-old) |
|---|---:|---:|---:|
| Gates detected | 5.333 | 5.000 | +0.333 |
| Trajectory coverage | 0.938 | 0.938 | +0.000 |
| Physics issue count | 5.000 | 5.000 | +0.000 |
| P90 speed (km/h) | 22.688 | 22.614 | +0.075 |
| Max speed (km/h) | 1516.597 | 389.780 | +1126.817 |
| Max G-force | 847.875 | 512.105 | +335.769 |
| Max jump (m) | 14.350 | 4.983 | +9.367 |
| Auto-cal correction | 84.777 | 83.781 | +0.996 |

## Verdict
- Gate detection count is mixed on these 3 videos (new better on 1, old better on 1, tie on 1).
- End-to-end stability is worse with the new model in most outlier metrics here:
  - max speed: old better on 2/3 videos
  - max G-force: old better on 3/3 videos
  - max jump: old better on 2/3 videos
- Both models fail physics validation on all 3 videos under current pipeline settings.
- Practical conclusion: keep the new detector as the best static holdout model, but **do not treat it as an end-to-end pipeline upgrade yet** without additional stabilization tuning/data fixes.

## Artifacts
- Summary JSON: `docs/reports/video_regression_20260214_summary.json`
- New model outputs: `artifacts/regression_20260214/new_model/`
- Old model outputs: `artifacts/regression_20260214/old_model/`
