# Dynamic Scale Ablation (2026-02-15)

## Setup
- Videos: 2907, 2909, 2911 regression set
- Shared settings: `stabilize=true`, `projection=scale`, `gate_spacing=8`, `camera_pitch=6`, `discipline=slalom`
- Comparison: Phase 4 dynamic scale ON vs OFF

## Default Spacing Update
- Pipeline constructor now uses discipline-aware defaults when `gate_spacing_m` is omitted:
  - `slalom`: `9.5`
  - `giant_slalom`: `27.0`
- Explicit `gate_spacing_m` still overrides defaults.

## Per-Video Comparison
| video | correction ON | correction OFF | p90 ON (km/h) | p90 OFF (km/h) | unable ON | unable OFF |
|---|---:|---:|---:|---:|---:|---:|
| 2907 | 3.45 | 3.39 | 189.9 | 186.5 | false | false |
| 2909 | 3.37 | 3.23 | 185.4 | 177.7 | false | false |
| 2911 | 4.25 | 3.72 | 233.8 | 204.7 | false | false |

## Aggregate
| mode | mean correction | max correction | mean p90 km/h | unable-to-calibrate rate |
|---|---:|---:|---:|---:|
| dynamic ON | 3.69 | 4.25 | 203.1 | 0.0% |
| dynamic OFF | 3.45 | 3.72 | 189.6 | 0.0% |

## Interpretation
- Turning dynamic scale OFF reduced mean correction by 0.24x (3.69 -> 3.45).
- Turning dynamic scale OFF reduced mean pre-auto-cal P90 by 13.5 km/h (203.1 -> 189.6).
- Dynamic scale contributes to scale inflation in this configuration, but disabling it does not bring P90 into the target 15–70 km/h range, so additional upstream geometry issues remain.

## Artifacts
- Raw rows: `tracks/C_geometry_scale/reports/geometry_dynamic_scale_ablation_20260215.json`
- Dynamic ON outputs: `/tmp/cgeom_verify_gs8`
- Dynamic OFF outputs: `/tmp/cgeom_verify_gs8_nodyn`
