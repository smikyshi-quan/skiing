# Results analysis: `/Users/quan/Documents/personal/Stanford application project/tests/test_videos_result_2026-02-24_translation`

Generated: 2026-02-24T21:59:16

## Summary
- Videos: 11 (ok=7, error=4)
- Physics: pass=0, fail=11
- Gates < 2: 4
- Missing analysis JSONs: 4
- Mixed-version flags: 0

## Table
| video | status | timestamp | commit | gates | track | cov | physics | p90_kmh | vmax_kmh | max_jump_m | j>10m | j>50m | auto_calib | failure | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMG_1478.MOV | ok | 2026-02-24T21:55:0… | 64145601 | 3 | bytetrack | 0.991 | FAIL | 55.00 | 140.18 | 0.76 | 0 | 0 | applied (3.12x) |  |  |
| 28_1752484118(原视频).mp4 | ok | 2026-02-24T21:22:0… | 64145601 | 3 | bytetrack | 1.000 | FAIL | 79.39 | 96.19 | 0.92 | 0 | 0 | applied (2.12x) |  |  |
| 30_1752484596(原视频).mp4 | ok | 2026-02-24T21:24:0… | 64145601 | 3 | bytetrack | 1.000 | FAIL | 79.61 | 97.92 | 0.45 | 0 | 0 | applied (2.66x) |  |  |
| 38_1752843425(原视频).mp4 | ok | 2026-02-24T21:25:1… | 64145601 | 2 | bytetrack | 0.993 | FAIL | 80.00 | 105.35 | 2.27 | 0 | 0 | applied (2.05x) |  |  |
| 594_1732936638(原视频).mp4 | ok | 2026-02-24T21:27:4… | 64145601 | 2 | bytetrack | 0.999 | FAIL | 80.00 | 81.29 | 0.74 | 0 | 0 | applied (3.10x) |  |  |
| IMG_1309.MOV | ok | 2026-02-24T21:52:3… | 64145601 | 2 | bytetrack | 0.999 | FAIL | 78.90 | 125.37 | 0.58 | 0 | 0 | applied (2.87x) |  |  |
| mmexport1704088159935.mp4 | error |  |  | 0 |  |  |  |  |  |  | 0 | 0 |  | Insufficient distinct gates for scale/projection. Detected … | no_analysis_json |
| mmexport1704089261026.mp4 | error |  |  | 0 |  |  |  |  |  |  | 0 | 0 |  | Insufficient distinct gates for scale/projection. Detected … | no_analysis_json |
| mmexport1706098456374.mp4 | error |  |  | 0 |  |  |  |  |  |  | 0 | 0 |  | Insufficient distinct gates for scale/projection. Detected … | no_analysis_json |
| 长城岭12.12(原视频).mp4 | ok | 2026-02-24T21:32:0… | 64145601 | 2 | bytetrack | 0.995 | FAIL | 79.80 | 135.17 | 1.22 | 0 | 0 | applied (1.71x) |  |  |
| 长城岭12.8.mp4 | error |  |  | 0 |  |  |  |  |  |  | 0 | 0 |  | Insufficient distinct gates for scale/projection. Detected … | no_analysis_json |

## Worst discontinuities (top 5 jumps per video)

### IMG_1478.MOV
- 0.76m jump: frame 1204 → 1207
- 0.65m jump: frame 1182 → 1183
- 0.65m jump: frame 1184 → 1185
- 0.65m jump: frame 1178 → 1179
- 0.65m jump: frame 1180 → 1181

### 28_1752484118(原视频).mp4
- 0.92m jump: frame 83 → 84
- 0.92m jump: frame 82 → 83
- 0.89m jump: frame 372 → 373
- 0.89m jump: frame 370 → 371
- 0.89m jump: frame 369 → 370

### 30_1752484596(原视频).mp4
- 0.45m jump: frame 390 → 391
- 0.45m jump: frame 396 → 397
- 0.45m jump: frame 392 → 393
- 0.45m jump: frame 393 → 394
- 0.45m jump: frame 394 → 395

### 38_1752843425(原视频).mp4
- 2.27m jump: frame 477 → 480
- 2.22m jump: frame 569 → 572
- 1.48m jump: frame 604 → 606
- 0.98m jump: frame 165 → 166
- 0.97m jump: frame 167 → 168

### 594_1732936638(原视频).mp4
- 0.74m jump: frame 1290 → 1292
- 0.74m jump: frame 1516 → 1518
- 0.38m jump: frame 523 → 524
- 0.38m jump: frame 530 → 531
- 0.38m jump: frame 520 → 521

### IMG_1309.MOV
- 0.58m jump: frame 83 → 84
- 0.58m jump: frame 78 → 79
- 0.58m jump: frame 79 → 80
- 0.58m jump: frame 81 → 82
- 0.58m jump: frame 85 → 86

### 长城岭12.12(原视频).mp4
- 1.22m jump: frame 373 → 375
- 1.16m jump: frame 6 → 10
- 0.63m jump: frame 521 → 522
- 0.63m jump: frame 520 → 521
- 0.63m jump: frame 522 → 523
