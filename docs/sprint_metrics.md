# Sprint Graduation Criteria

**Phase**: 2 — Detection/Tracking Quality Sprint
**Last updated**: 2026-02-26
**Gating**: All criteria must pass before Phase 3 (reintroduce 3D) begins.

---

## Pass/Fail Criteria

| # | Criterion | Target | Current (2026-02-26) | Status |
|---|-----------|--------|----------------------|--------|
| G1 | Gate holdout F1 @ conf=0.35 | ≥ 0.85 | 0.630* | ❌ FAIL |
| G2 | `track_id_switches` mean (regression suite) | ≤ 2 | 0.00 | ✅ PASS |
| G3 | `trajectory_coverage` mean (regression suite) | ≥ 0.80 | 0.988 | ✅ PASS |
| G4 | No Stage 2 metric regression > 20% vs. baseline | — | TBD | ⏳ PENDING |

*F1 estimate: P=0.615, R=0.645 → F1 = 2·P·R/(P+R) ≈ 0.630. Measured on valid split of `final_combined_1class_20260226_curated`.

---

## Measurement Method

### G1 — Gate holdout F1
```bash
python scripts/train_detector.py \
  --data data/datasets/final_combined_1class_20260226_curated/data.yaml \
  --eval-only models/gate_detector_best.pt \
  --save-metrics
```
F1 = 2·Precision·Recall / (Precision + Recall) at conf=0.35.

### G2 — Track ID switches
Measured from `track_id_switches` field in pipeline JSON output, averaged across
regression videos 2907, 2909, 2911. See `tracks/D_tracking_outlier/reports/tracker_sweep_*.json`.

### G3 — Trajectory coverage
`len(trajectory_2d) / video_info.total_frames` averaged across regression suite.

### G4 — No Stage 2 regression > 20%
Run `scripts/run_eval.py --model <new_model> --baseline tracks/E_evaluation_ci/reports/baseline_regression.json`.
Stage 3 must emit PASS.

---

## Phase 2 Findings (2026-02-26)

### Task 2.1 — Data Curation
- **Dataset**: `data/datasets/final_combined_1class_20260226_curated`
- 260 annotated train images + **100 hard-negatives** (50 restored from orig, 50 new from regression videos)
- 3 `h≥1.0` bboxes fixed (clamped to 0.98)
- Hard-negative ratio: 28% (target: 30–40%) ✓

### Task 2.2 — Gate Detector Retraining
- **Baseline model**: `models/gate_detector_best.pt`
  - mAP@0.5 = 0.615, P = 0.615, R = 0.645, F1 ≈ 0.630
- **New training run**: `runs/detect/gate_detector_20260226_*/`
  - Training: YOLOv8s, imgsz=960, batch=8, freeze=10, cos-lr, 150 epochs
  - Dataset: `final_combined_1class_20260226_curated` (360 train, 72 valid, 26 test)
  - Status: ⏳ training in background
- **Promotion criteria**: new model mAP@0.5 > 0.615 AND no Stage 2 regression > 20%
- **Action on promotion**: copy `best.pt` → `models/gate_detector_best.pt`, update `shared/docs/MODEL_REGISTRY.md`

### Task 2.3 — Tracker Threshold Sweep
Results from `tracks/D_tracking_outlier/reports/tracker_sweep_20260226_*.json`:

| skier_conf | mean coverage | mean switches | verdict |
|------------|---------------|---------------|---------|
| 0.15 | 0.9883 | 0.00 | ✅ BEST coverage |
| 0.20 | 0.9863 | 0.00 | ✅ |
| **0.25** | **0.9846** | **0.00** | **← current default** |
| 0.30 | 0.9791 | 0.00 | ✓ |
| 0.35 | 0.9719 | 0.00 | ✓ |

**Finding**: Track ID switches = 0 across all values. Coverage decreases monotonically.
**Recommendation**: lower `skier_conf` to **0.15** for +1.4pp coverage gain (0.9846 → 0.9883).
**Config update**: `skier_conf: 0.15` in `configs/regression_defaults.yaml` ← pending 2.4 completion.

### Task 2.4 — Gate Recall Tuning
Results from `tracks/D_tracking_outlier/reports/gate_recall_sweep_20260226_1025.json`:

**gate_search_frames sweep** (gate_track_frames=120, skier_conf=0.15):

| gate_search_frames | mean gates | mean coverage |
|--------------------|------------|---------------|
| 100 | 3.67 | 0.9883 |
| 150 | 3.67 | 0.9883 |
| 200 | 3.67 | 0.9883 |
| 300 | 3.67 | 0.9883 |

**gate_track_frames sweep** (gate_search_frames=300, skier_conf=0.15):

| gate_track_frames | mean gates | mean coverage |
|-------------------|------------|---------------|
| 0 | 3.67 | 0.9883 |
| 60 | 3.67 | 0.9883 |
| 120 | 3.67 | 0.9883 |
| 240 | 3.67 | 0.9883 |

**Finding**: Gates on regression videos are detected in the first 100 frames — neither `gate_search_frames` nor `gate_track_frames` affects gate count or coverage on these videos. The gate bottleneck is **model recall**, not search budget.

**Recommendation**: Keep both at their current defaults (`gate_search_frames: 150`, `gate_track_frames: 120`). Budget doesn't help without a better detector.

**Config update**: Only `skier_conf: 0.15` (from 2.3) written to `configs/regression_defaults.yaml`.

---

## Phase 3 Readiness Checklist

- [ ] G1: Gate F1 ≥ 0.85 — **blocked on retraining completion**
- [x] G2: Track ID switches ≤ 2 — PASS (= 0)
- [x] G3: Coverage ≥ 0.80 — PASS (= 0.988)
- [ ] G4: No Stage 2 regression > 20% — pending new model eval
- [ ] Update `configs/regression_defaults.yaml` with best sweep params
- [ ] Update `shared/docs/MODEL_REGISTRY.md` with new model checkpoint

**Phase 3 is blocked until G1 passes (new model must achieve F1 ≥ 0.85).**
