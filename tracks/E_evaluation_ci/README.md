# Track E: Evaluation / CI Pipeline

## Owner
Assigned to: _______________

## Goal
Build a single-command evaluation pipeline that all other tracks use to measure progress.

**THIS TRACK SHOULD START FIRST.** Every other track depends on it.

## Your Spec Document
**Read this first:** `Track_E_Evaluation_CI_Pipeline.docx` (in this folder)

## Files You Create (NEW)
| File | What it does |
|------|-------------|
| `scripts/run_eval.py` | **Main deliverable** — single command evaluation |
| `configs/regression_defaults.yaml` | Frozen pipeline parameters for regression |

## Files You Extend (edit these)
| File | What to change |
|------|---------------|
| `scripts/evaluate.py` | Add CLI args for model path, data path, JSON output |

## Frozen Inputs (do NOT modify)
| Resource | Location |
|----------|----------|
| Regression videos | `tracks/E_evaluation_ci/regression_videos/` (3 videos: 2907, 2909, 2911) |
| Test dataset | `data/annotations/final_combined_1class_20260213/test/` |
| Current best model | `models/gate_detector_best.pt` |

## Your Reports Folder
`tracks/E_evaluation_ci/reports/` — contains baseline regression results for comparison.

## What Your Script Must Produce
```
docs/reports/eval_YYYYMMDD_HHMM/
  stage1_holdout.json        # Gate detection P/R/F1
  stage2_regression.json     # 3-video pipeline metrics
  stage2_regression/
    2907_analysis.json
    2909_analysis.json
    2911_analysis.json
  summary.md                 # Human-readable combined report
  eval_result.json           # PASS/FAIL verdict
```

## The One Command
```bash
python scripts/run_eval.py \
  --model models/gate_detector_best.pt \
  --baseline docs/reports/eval_baseline.json
```

## Dependencies
- **No upstream dependencies** — start immediately
- **All other tracks depend on you**
