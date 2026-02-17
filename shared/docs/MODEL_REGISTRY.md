# Model Registry

This file tracks the currently promoted gate detector checkpoint used by the pipeline.

## Active Model
- Alias: `models/gate_detector_best.pt`
- Promoted on: `2026-02-14`
- Source run: `runs/detect/gate_detector_20260214_0957/weights/best.pt`
- SHA256: `8968fb4f153a18b793eb1190aebf19d4df30db48064c1a62c912b7a49314f7ac`
- Size: `6,319,338 bytes`
- Notes: Early-stopped at epoch 85. Stronger Precision/Recall/mAP than prior run.

## Active Inference Defaults
- Gate confidence (`gate_conf`): `0.35`
- Gate NMS IoU (`gate_iou`): `0.55`
- Reason: best balanced F1 on holdout threshold sweep for current model.

## Rollback Model
- Backup path: `models/gate_detector_best_20260213_2337_backup.pt`
- Source run: `runs/detect/gate_detector_20260213_2337/weights/best.pt`
- SHA256: `da67e58315c885f401cbb4dffa14467268e0f4f07051dd3133f7a8e52d4ae6f7`
- Size: `22,515,242 bytes`

## Related Evaluation Notes
- Comparison report: `docs/reports/model_comparison_20260214_0957_vs_20260213_2337.md`
- Holdout test report: `docs/reports/holdout_test_comparison_20260214_0957_vs_20260213_2337.md`
- Threshold tuning report: `docs/reports/inference_threshold_tuning_20260214.md`


## Candidate Cycle 2026-02-15 (Not Promoted)
- Checkpoint: `models/gate_detector_best_20260215_1043.pt`
- Source run: `tracks/B_model_retraining/runs/detect/gate_detector_20260215_1016/weights/best.pt`
- SHA256: `cd791cd8ff5fd70a3339dc87557cb183b49a84f5bd58d8109d8b37a4eb3887ce`
- Size: `6,311,082 bytes`
- Calibration sweep best: `gate_conf=0.35`, `gate_iou=0.55`, `F1=0.5000` on calibration split
- Frozen test (23 images, 54 instances): `P=0.7143`, `R=0.6481`, `F1=0.6796`, `TP/FP/FN=35/14/19`
- Regression summary: large reductions in max speed/max G/max jump on all three videos, but Stage 1 F1 regressed versus baseline (`0.8727 -> 0.6796`)
- Decision: **Not promoted** (active model unchanged)

## Additional Reports (2026-02-15)
- Threshold sweep: `tracks/B_model_retraining/reports/threshold_sweep_20260215.json`
- Frozen test eval: `tracks/B_model_retraining/reports/eval_20260215.json`
- Regression comparison: `tracks/B_model_retraining/reports/regression_20260215.md`
- Full eval run: `tracks/B_model_retraining/reports/eval_20260215_1045/eval_result.json`
