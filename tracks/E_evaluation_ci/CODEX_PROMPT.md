# Codex Prompt for Track E — Evaluation / CI Pipeline

Paste everything below into the Codex thread that has access to `tracks/E_evaluation_ci/`.

---

Build a single-command evaluation pipeline for our alpine ski racing video analysis project. This is the most important infrastructure — every other team member needs it to measure their work.

## Context

We have a computer vision pipeline in `ski_racing/` that detects ski gates (YOLOv8), tracks the skier (ByteTrack), projects 2D→3D coordinates, and validates physics. The gate detector currently scores F1=0.755 (P=0.75, R=0.76) on 38 test images with 138 gate instances. The full pipeline FAILS physics validation on all test videos — speeds reach 3000+ km/h, G-forces reach 1400+G. Other people are fixing those problems separately. Your job is to build the measurement tool.

## What to build

Create `scripts/run_eval.py` — one command, three stages:

**Stage 1 — Holdout Detection Metrics:**
Run the gate detector on every image in `data/annotations/final_combined_1class_20260213/test/`. Match predictions to ground truth using IoU >= 0.50 greedy matching. Output precision, recall, F1, TP, FP, FN as JSON. Include breakdown at confidence thresholds 0.25, 0.35, 0.45, 0.55.

Look at `scripts/evaluate.py` for existing evaluation logic — extend it to accept `--model`, `--data`, `--output` CLI args and produce structured JSON.

**Stage 2 — 3-Video Regression:**
Run the full pipeline (`ski_racing/pipeline.py` → `SkiRacingPipeline`) on 3 frozen regression videos already in `tracks/E_evaluation_ci/regression_videos/` (files: 2907, 2909, 2911). Use these frozen settings — create them as `configs/regression_defaults.yaml`:
- gate_conf=0.35, gate_iou=0.55
- stabilize=True, camera_mode="affine", projection="scale"
- discipline="slalom", gate_spacing_m=12.0

For each video, extract: gates detected, trajectory coverage (fraction of frames with tracking), P90 speed (km/h), max speed (km/h), max G-force, max jump (meters), auto-cal correction factor, physics issue count. These are all available in the pipeline's output JSON.

Look at `scripts/process_video.py` to see how the pipeline is currently invoked from CLI.

**Stage 3 — Summary Report:**
Combine Stage 1 + Stage 2 into `summary.md`. If `--baseline` is provided (path to a previous `eval_result.json`), generate a delta table. Print a PASS/FAIL verdict: PASS if F1 >= baseline F1 AND no regression metric degrades >20%. FAIL otherwise, listing specific reasons.

## Usage

```bash
python scripts/run_eval.py --model models/gate_detector_best.pt
python scripts/run_eval.py --model models/gate_detector_best.pt --baseline docs/reports/eval_baseline.json
```

## Output structure

```
docs/reports/eval_YYYYMMDD_HHMM/
  stage1_holdout.json
  stage2_regression.json
  stage2_regression/
    2907_analysis.json
    2909_analysis.json
    2911_analysis.json
  summary.md
  eval_result.json        # PASS/FAIL + all metrics combined
```

## Files to read first

1. `tracks/E_evaluation_ci/Track_E_Evaluation_CI_Pipeline.docx` — full spec with acceptance criteria
2. `scripts/evaluate.py` — existing eval code
3. `scripts/process_video.py` — existing pipeline CLI
4. `ski_racing/pipeline.py` — understand `SkiRacingPipeline.process_video()` return format
5. `tracks/E_evaluation_ci/reports/baseline_regression.md` — current numbers to expect

## Files you create

- `scripts/run_eval.py` (NEW — main deliverable)
- `configs/regression_defaults.yaml` (NEW)

## Files you extend

- `scripts/evaluate.py` (add CLI args, JSON output)

## Do NOT modify

- `tracks/E_evaluation_ci/regression_videos/` (frozen)
- `data/annotations/final_combined_1class_20260213/test/` (frozen)
- `ski_racing/*.py` (other people's responsibility)

## How to verify your work

1. Run your script twice. Both runs must produce identical numbers.
2. Run with the backup model: `models/gate_detector_best_20260213_2337_backup.pt`. It should produce a FAIL verdict.
3. Total runtime should be under 20 minutes.
