# Manager Dispatch: Worker E START

Timestamp: 2026-02-14 20:56:01
Priority: FIRST (highest)
Status: ACTIVE

Use the exact prompt below.

```text
You are working on an alpine ski racing video analysis project.
Your job: build a single-command evaluation pipeline that all other team members
will use to measure whether their code changes helped or hurt.
 
== PROJECT CONTEXT ==
We have a computer vision pipeline that: detects ski gates (YOLOv8), tracks the skier
(ByteTrack), projects 2D pixel coords to 3D meters, and validates physics (speed, G-force).
Current gate detection: F1=0.755, P=0.75, R=0.76 on 38 test images (138 gate instances).
The pipeline FAILS physics validation on all test videos (speeds 3000+ km/h, G-forces 1400+G).
Other people are fixing those problems. YOUR job: build the measurement tool.
 
== YOUR DELIVERABLE: scripts/run_eval.py ==
One command that runs three stages:
 
Stage 1 - Holdout Detection Metrics:
  Run the gate detector on data/annotations/final_combined_1class_20260213/test/
  Match detections to ground truth (IoU >= 0.50). Output: P, R, F1, TP, FP, FN as JSON.
 
Stage 2 - 3-Video Regression:
  Run the full pipeline (ski_racing/pipeline.py) on 3 frozen videos in
  tracks/E_evaluation_ci/regression_videos/ (videos: 2907, 2909, 2911).
  Pipeline settings are frozen in configs/regression_defaults.yaml (you create this).
  Settings: gate_conf=0.35, gate_iou=0.55, stabilize=True, camera_mode=affine,
  projection=scale, discipline=slalom.
  Output per video: gates detected, trajectory coverage, P90 speed, max speed,
  max G-force, max jump (m), auto-cal correction factor, physics issue count.
 
Stage 3 - Summary Report:
  Combine Stage 1 + Stage 2 into a summary.md with PASS/FAIL verdict.
  PASS if: F1 >= baseline AND no regression metric degrades >20%.
  FAIL otherwise, listing specific failure reasons.
 
Usage: python scripts/run_eval.py --model models/gate_detector_best.pt \
         --baseline docs/reports/eval_baseline.json
 
Output structure:
  docs/reports/eval_YYYYMMDD_HHMM/
    stage1_holdout.json
    stage2_regression.json
    stage2_regression/2907_analysis.json, 2909_analysis.json, 2911_analysis.json
    summary.md
    eval_result.json  (PASS/FAIL + all metrics)
 
== FILES TO READ FIRST ==
1. tracks/E_evaluation_ci/Track_E_Evaluation_CI_Pipeline.docx  (full spec)
2. tracks/E_evaluation_ci/README.md  (quick reference)
3. scripts/evaluate.py  (existing eval code - extend this)
4. scripts/process_video.py  (existing pipeline CLI)
5. ski_racing/pipeline.py  (understand SkiRacingPipeline.process_video() interface)
6. tracks/E_evaluation_ci/reports/baseline_regression.md  (current numbers)
 
== FILES YOU CREATE ==
- scripts/run_eval.py  (NEW - main deliverable)
- configs/regression_defaults.yaml  (NEW - frozen settings)
 
== FILES YOU EXTEND ==
- scripts/evaluate.py  (add CLI args, JSON output)
 
== DO NOT MODIFY ==
- tracks/E_evaluation_ci/regression_videos/  (frozen test videos)
- data/annotations/final_combined_1class_20260213/test/  (frozen test set)
- ski_racing/*.py  (not your responsibility)
 
== ACCEPTANCE TEST ==
1. Run your script twice consecutively. Both must produce identical numbers.
2. Run with backup model (models/gate_detector_best_20260213_2337_backup.pt).
   It must produce FAIL (since that model is worse).
3. Total runtime < 20 minutes for all 3 stages.
```
