# Kalman Tuning (20260214)

- Videos: `2907`, `2909`, `2911`
- Q sweep: `[200, 400, 800, 1600, 3200]`
- Recommended Q (aggregate): **200**

## Aggregate Across 3 Videos

| Q | Mean RMS diff (px) | Mean max speed (km/h) | Mean max G | Mean max jump (px) |
|---|---:|---:|---:|---:|
| 200 | 926.77 | 10435.06 | 4288.06 | 30.83 |
| 400 | 153.89 | 12135.01 | 6412.28 | 12.44 |
| 800 | 122.00 | 12157.76 | 5575.57 | 23.02 |
| 1600 | 93.91 | 12235.46 | 5136.71 | 51.62 |
| 3200 | 74.58 | 12235.54 | 5117.77 | 79.12 |

## Video 2907

| Q | RMS diff (px) | Max speed (km/h) | Max G | Max jump (px) | ByteTrack coverage | Track-ID switches | Outlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | 203.51 | 21834.01 | 2339.21 | 9.27 | 0.989 | 1 | 0.829% |
| 400 | 124.29 | 21851.81 | 2820.37 | 8.37 | 0.989 | 1 | 0.829% |
| 800 | 117.05 | 21868.87 | 2124.86 | 11.97 | 0.989 | 1 | 0.829% |
| 1600 | 107.88 | 21879.38 | 2070.66 | 85.38 | 0.989 | 1 | 0.829% |
| 3200 | 17.88 | 21882.22 | 2089.26 | 81.29 | 0.989 | 1 | 0.829% |

## Video 2909

| Q | RMS diff (px) | Max speed (km/h) | Max G | Max jump (px) | ByteTrack coverage | Track-ID switches | Outlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | 134.55 | 1344.76 | 1956.76 | 5.18 | 0.997 | 5 | 0.220% |
| 400 | 131.53 | 5789.61 | 8491.33 | 10.03 | 0.997 | 5 | 0.220% |
| 800 | 90.19 | 5839.03 | 8510.91 | 13.15 | 0.997 | 5 | 0.220% |
| 1600 | 53.18 | 6064.41 | 8518.07 | 22.11 | 0.997 | 5 | 0.220% |
| 3200 | 49.22 | 6065.24 | 8526.96 | 32.32 | 0.997 | 5 | 0.220% |

## Video 2911

| Q | RMS diff (px) | Max speed (km/h) | Max G | Max jump (px) | ByteTrack coverage | Track-ID switches | Outlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | 2442.26 | 8126.42 | 8568.22 | 78.05 | 0.969 | 5 | 0.901% |
| 400 | 205.83 | 8763.60 | 7925.15 | 18.92 | 0.969 | 5 | 0.901% |
| 800 | 158.77 | 8765.39 | 6090.95 | 43.93 | 0.969 | 5 | 0.901% |
| 1600 | 120.68 | 8762.59 | 4821.40 | 47.38 | 0.969 | 5 | 0.901% |
| 3200 | 156.64 | 8759.17 | 4737.08 | 123.76 | 0.969 | 5 | 0.901% |

## Notes

- Physics extremes remain dominated by geometry/scale instability; Track D changes improved 2D continuity and coverage but cannot fully fix 3D explosions alone.
- Raw per-run metrics are saved in the JSON companion report.

JSON artifact: `/Users/quan/Documents/personal/Stanford application project/tracks/D_tracking_outlier/reports/kalman_tuning_20260214.json`
