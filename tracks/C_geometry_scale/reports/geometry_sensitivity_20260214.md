# Geometry Sensitivity Sweep (2026-02-14)

## Setup
- Videos: 2907, 2909, 2911 (`tracks/E_evaluation_ci/regression_videos/`)
- Model: `models/gate_detector_best.pt`
- Dynamic scale smoothing/bounds enabled (median window 15, reject outside [0.5x, 2.0x] median, interpolate rejects)
- Auto-cal guard rails enabled (warning >3x, fail calibration >5x)
- Coordination note: no pipeline phase reordering change was introduced in this Track C pass.

## Gate Spacing Sweep (stabilize=true, projection=scale, camera_pitch=6)
| gate_spacing_m | mean correction | max correction | mean P90 km/h (pre-auto-cal) | mean max speed km/h | mean max G | unable-to-calibrate rate |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3.69 | 4.25 | 203.1 | 2250.3 | 1016.0 | 0.0% |
| 10 | 4.62 | 5.31 | 253.8 | 4227.0 | 2389.6 | 33.3% |
| 12 | 5.54 | 6.38 | 304.6 | 12157.8 | 5575.6 | 100.0% |
| 14 | 6.46 | 7.44 | 355.4 | 14184.1 | 6504.8 | 100.0% |

## Camera Pitch Sweep (stabilize=true, projection=scale, gate_spacing=12)
| camera_pitch_deg | mean correction | max correction | mean P90 km/h (pre-auto-cal) | mean max speed km/h | mean max G | unable-to-calibrate rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52.98 | 61.01 | 2914.1 | 116310.5 | 53340.2 | 100.0% |
| 10 | 9.20 | 10.59 | 506.0 | 20197.1 | 9262.4 | 100.0% |
| 20 | 18.12 | 20.87 | 996.7 | 39780.5 | 18243.4 | 100.0% |
| 30 | 26.49 | 30.51 | 1457.0 | 58155.3 | 26670.1 | 100.0% |

## Projection Comparison
- `stabilize=true` currently forces requested `projection=homography` to effective `projection=scale` (guard rail in pipeline), so stabilized projection comparison is not currently meaningful.
- To compare projection behavior directly, runs below use `stabilize=false` (same gate spacing/pitch/model).

| mode (stabilize=false) | mean correction | max correction | mean P90 km/h (pre-auto-cal) | mean max speed km/h | mean max G | unable-to-calibrate rate |
|---|---:|---:|---:|---:|---:|---:|
| scale | 1.79 | 2.60 | 100.9 | 777.0 | 1046.5 | 0.0% |
| homography | 1.17 | 1.51 | 55.8 | 1857.1 | 782.8 | 0.0% |

## Best Observed Stabilized Config (from tested sweep)
- Best by correction factor: `gate_spacing_m=8`, `camera_pitch_deg=6`, `projection=scale`, `stabilize=true`
- Mean correction factor: 3.69
- Max correction factor: 4.25
- Mean pre-auto-cal P90 speed: 203.1 km/h

## Pass-Criteria Check (Best Observed Config)
- Correction mean < 5x: PASS (3.69)
- Correction max < 10x: PASS (4.25)
- P90 pre-auto-cal in 15-70 km/h: FAIL (203.1)

## Artifacts
- Real-run raw rows: `tracks/C_geometry_scale/reports/geometry_sensitivity_20260214_real_runs.json`
- Output directories used:
  - `/tmp/cgeom_verify_gs8` (gs8_p6_scale_stab)
  - `/tmp/cgeom_verify_gs10` (gs10_p6_scale_stab)
  - `/tmp/cgeom_verify_default` (gs12_p6_scale_stab)
  - `/tmp/cgeom_verify_gs14` (gs14_p6_scale_stab)
  - `/tmp/cgeom_verify_pitch0` (gs12_p0_scale_stab)
  - `/tmp/cgeom_verify_pitch10` (gs12_p10_scale_stab)
  - `/tmp/cgeom_verify_pitch20` (gs12_p20_scale_stab)
  - `/tmp/cgeom_verify_pitch30` (gs12_p30_scale_stab)
  - `/tmp/cgeom_verify_homography` (req_homography_stab)
  - `/tmp/cgeom_verify_proj_scale_nostab` (scale_nostab)
  - `/tmp/cgeom_verify_proj_homography_nostab` (homography_nostab)
