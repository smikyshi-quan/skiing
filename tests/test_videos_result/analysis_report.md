# Results analysis: `/Users/quan/Documents/personal/Stanford application project/tests/test_videos_result`

Generated: 2026-02-24T21:59:28

## Summary
- Videos: 11 (ok=11, error=0)
- Physics: pass=0, fail=11
- Gates < 2: 4
- Missing analysis JSONs: 0
- Mixed-version flags: 11

## Table
| video | status | timestamp | commit | gates | track | cov | physics | p90_kmh | vmax_kmh | max_jump_m | j>10m | j>50m | auto_calib | failure | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 594_1732936638(原视频).mp4 | ok | 2026-02-20T22:19:4… |  | 2 | bytetrack | 0.999 | FAIL | 147.95 | 29600.28 | 138.45 | 71 | 49 | unable (5.52x, correcti… |  | missing_git_metadata |
| IMG_1309.MOV | ok | 2026-02-21T00:38:3… |  | 2 | bytetrack | 0.999 | FAIL | 75.47 | 628.93 | 3.27 | 0 | 0 | applied (3.09x) |  | missing_git_metadata |
| IMG_1478.MOV | ok | 2026-02-21T01:53:0… |  | 3 | bytetrack | 0.991 | FAIL | 168.22 | 13113.99 | 63.13 | 72 | 11 | unable (6.93x, correcti… |  | missing_git_metadata |
| mmexport1704088159935.mp4 | ok | 2026-02-21T02:51:4… |  | 1 | bytetrack | 0.969 | FAIL |  |  | 17.53 | 6 | 0 | unable (15.78x, correct… |  | missing_git_metadata |
| mmexport1704089261026.mp4 | ok | 2026-02-21T03:49:4… |  | 1 | bytetrack | 0.994 | FAIL |  |  | 41.43 | 18 | 0 | unable (6.90x, correcti… |  | missing_git_metadata |
| mmexport1706098456374.mp4 | ok | 2026-02-21T04:00:2… |  | 1 | bytetrack | 0.919 | FAIL |  |  | 69.47 | 97 | 4 | unable (17.18x, correct… |  | missing_git_metadata |
| 28_1752484118(原视频).mp4 | ok | 2026-02-20T21:51:2… |  | 3 | temporal | 0.000 | FAIL | 192.71 | 5513.25 | 54.81 | 41 | 1 | unable (5.35x, correcti… | ModuleNotFoundError: No module named 'lap' | missing_git_metadata, old_tempo… |
| 30_1752484596(原视频).mp4 | ok | 2026-02-20T21:57:2… |  | 3 | temporal | 0.000 | FAIL | 52.50 | 440.32 | 2.30 | 0 | 0 | applied (4.24x) | ModuleNotFoundError: No module named 'lap' | missing_git_metadata, old_tempo… |
| 38_1752843425(原视频).mp4 | ok | 2026-02-20T22:04:0… |  | 2 | bytetrack | 0.993 | FAIL | 80.00 | 283.96 | 5.02 | 0 | 0 | applied (2.05x) |  | missing_git_metadata |
| 长城岭12.12(原视频).mp4 | ok | 2026-02-21T05:14:4… |  | 2 | bytetrack | 0.995 | FAIL | 80.00 | 80.00 | 1.47 | 0 | 0 | applied (1.40x) |  | missing_git_metadata |
| 长城岭12.8.mp4 | ok | 2026-02-22T19:21:0… |  | 1 | bytetrack | 0.871 | FAIL |  |  | 4.73 | 0 | 0 | applied (3.93x) |  | missing_git_metadata |

## Worst discontinuities (top 5 jumps per video)

### 594_1732936638(原视频).mp4
- 138.45m jump: frame 181 → 182
- 136.23m jump: frame 279 → 280
- 135.61m jump: frame 578 → 579
- 135.61m jump: frame 417 → 418
- 135.22m jump: frame 576 → 577

### IMG_1309.MOV
- 3.27m jump: frame 546 → 547
- 3.03m jump: frame 608 → 609
- 2.92m jump: frame 753 → 754
- 2.91m jump: frame 712 → 713
- 2.91m jump: frame 677 → 678

### IMG_1478.MOV
- 63.13m jump: frame 58 → 59
- 62.45m jump: frame 57 → 58
- 62.16m jump: frame 65 → 66
- 61.49m jump: frame 27 → 28
- 60.96m jump: frame 26 → 27

### mmexport1704088159935.mp4
- 17.53m jump: frame 1677 → 1680
- 17.53m jump: frame 1681 → 1684
- 12.99m jump: frame 242 → 248
- 12.59m jump: frame 174 → 186
- 11.69m jump: frame 1596 → 1598

### mmexport1704089261026.mp4
- 41.43m jump: frame 424 → 425
- 40.52m jump: frame 390 → 391
- 40.30m jump: frame 428 → 429
- 40.15m jump: frame 430 → 431
- 39.11m jump: frame 427 → 428

### mmexport1706098456374.mp4
- 69.47m jump: frame 117 → 122
- 62.69m jump: frame 249 → 250
- 60.56m jump: frame 385 → 387
- 51.20m jump: frame 207 → 208
- 39.27m jump: frame 134 → 135

### 28_1752484118(原视频).mp4
- 54.81m jump: frame 242 → 243
- 48.66m jump: frame 292 → 293
- 43.86m jump: frame 238 → 239
- 40.80m jump: frame 237 → 238
- 40.42m jump: frame 229 → 230

### 30_1752484596(原视频).mp4
- 2.30m jump: frame 264 → 265
- 2.12m jump: frame 390 → 391
- 2.01m jump: frame 405 → 406
- 1.81m jump: frame 389 → 390
- 1.61m jump: frame 263 → 264

### 38_1752843425(原视频).mp4
- 5.02m jump: frame 8 → 9
- 4.17m jump: frame 6 → 7
- 3.66m jump: frame 28 → 29
- 3.59m jump: frame 11 → 12
- 3.53m jump: frame 27 → 28

### 长城岭12.12(原视频).mp4
- 1.47m jump: frame 6 → 10
- 0.74m jump: frame 349 → 351
- 0.74m jump: frame 373 → 375
- 0.74m jump: frame 343 → 345
- 0.74m jump: frame 364 → 366

### 长城岭12.8.mp4
- 4.73m jump: frame 214 → 223
- 3.68m jump: frame 369 → 372
- 3.62m jump: frame 465 → 466
- 2.94m jump: frame 521 → 522
- 2.80m jump: frame 444 → 445
