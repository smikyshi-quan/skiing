# Pipeline Ordering Study (20260214)

- Video: `2911_1765738746(Video in Original Quality).mp4`
- Kalman Q: `200`
- Compared:
  - Baseline: `track -> smooth -> transform(camera compensation)`
  - Alternative: `track -> camera compensate 2D -> smooth -> transform`

| Metric | Baseline | Alternative | Delta (alt-baseline) |
|---|---:|---:|---:|
| Max jump (px) | 78.05 | 23.18 | -54.88 |
| P95 jump (px) | 15.61 | 5.23 | -10.38 |
| Mean jump (px) | 10.78 | 2.86 | -7.92 |
| ByteTrack coverage | 0.97 | 0.97 | +0.00 |
| Track-ID switches | 0 | 0 | +0 |
| Outlier count | 7 | 68 | +61 |
| Max speed (km/h) | 8126.42 | 530.26 | -7596.16 |
| Max G | 8568.22 | 301.06 | -8267.15 |

## Recommendation

- Keep baseline ordering as default for now. Use the alternative ordering as an experiment flag pending Track C geometry stabilization alignment.
