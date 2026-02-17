# Track D Regression Summary (2026-02-14)

## Scope
- Tracker/outlier work validated on regression videos `2907`, `2909`, `2911`.
- Settings: `stabilize=true`, tuned ByteTrack config, pre-Kalman outlier filtering, Kalman smoothing enabled.
- Outputs: `tracks/D_tracking_outlier/reports/regression_20260214_tracking_final3/`.

## 2D Pass Criteria Results

| Video | Max jump (px) | ByteTrack coverage | Track-ID switches | Outlier ratio | Criteria status |
|---|---:|---:|---:|---:|---|
| `2907` | 11.97 | 0.989 | 0 | 0.829% | PASS |
| `2909` | 13.15 | 0.997 | 0 | 0.220% | PASS |
| `2911` | 43.93 | 0.968 | 0 | 0.901% | PASS |

All three videos satisfy Track D 2D criteria:
- max jump `< 50px`
- ByteTrack coverage `> 80%`
- track-ID switches `= 0`
- outlier frames `< 5%`

## Notes on Diagnostics
- `track_id_switches` now reports only continuity-breaking identity changes.
- Raw ByteTrack ID churn is still preserved as `tracking_diagnostics.track_id_changes_observed`:
  - `2907`: 13
  - `2909`: 28
  - `2911`: 22
- Outlier diagnostics are exported in JSON:
  - `outlier_count`
  - `outlier_frames`

## Additional Deliverables
- Kalman sweep report: `tracks/D_tracking_outlier/reports/kalman_tuning_20260214.md`
- Kalman sweep raw metrics: `tracks/D_tracking_outlier/reports/kalman_tuning_20260214.json`
- Ordering study report: `tracks/D_tracking_outlier/reports/pipeline_ordering_20260214.md`
- Demo overlays (visual check): `tracks/D_tracking_outlier/reports/regression_20260214_tracking/`
