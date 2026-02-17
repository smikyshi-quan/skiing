# Alpine Ski Racing AI — Track Workspaces

## Quick Start
1. Find your assigned track folder below
2. Read the README.md inside your track folder
3. Read the .docx spec document — it has all the details
4. Check the `reports/` subfolder for existing analysis and baselines

## Track Overview

| Track | Folder | Priority | Summary |
|-------|--------|----------|---------|
| **E** | `E_evaluation_ci/` | **Start first** | Build single-command eval pipeline. All tracks need this. |
| **B** | `B_model_retraining/` | High | Improve gate detection via data-centric retraining |
| **C** | `C_geometry_scale/` | High | Fix 2D→3D transform (auto-cal correction from 84x → <5x) |
| **D** | `D_tracking_outlier/` | High | Smooth 2D trajectories, fix tracker switches/jumps |
| **F** | `F_runtime_parallel/` | Low (last) | Speed optimization — only after B–E are stable |

## Dependency Graph
```
Track E (Eval/CI) ← no dependencies, START FIRST
    ↑
    ├── Track B (Retraining) — needs E to measure checkpoints
    │       ↑
    │       └── Track D (Tracking) — cleaner data helps retraining
    │
    ├── Track C (Geometry) — needs E to measure scale fixes
    │       ↕
    │       └── Track D (Tracking) — tightly coupled, coordinate
    │
    └── Track F (Parallelization) — needs B+C+D stable first
```

## Progress Summary (through 2026-02-17)

- **Track E (Eval/CI):** Single-command evaluation pipeline lives in `scripts/run_eval.py` with frozen defaults in `configs/regression_defaults.yaml`; baseline outputs and reference videos are under `tracks/E_evaluation_ci/`.
- **Track B (Retraining):** 2026-02-15 retraining + calibration cycle artifacts and eval outputs are under `tracks/B_model_retraining/artifacts/` and `tracks/B_model_retraining/reports/` (promotion decisions recorded in `shared/docs/MODEL_REGISTRY.md`).
- **Track C (Geometry/Scale):** Discipline auto-detect smoke test, geometry sensitivity, and dynamic-scale ablations are under `tracks/C_geometry_scale/reports/`.
- **Track D (Tracking/Outliers):** Kalman tuning, pipeline ordering study, and tracking regression runs (including overlay demos) are under `tracks/D_tracking_outlier/reports/`.
- **Track F (Runtime):** Profiling and perf artifacts are under `tracks/F_runtime_parallel/reports/`.

## Shared Resources (do NOT reorganize these)

| Resource | Location | Used by |
|----------|----------|---------|
| Source code | `ski_racing/` | All tracks (shared) |
| CLI scripts | `scripts/` | All tracks |
| Models | `models/` | All tracks |
| Training data | `data/annotations/` | Track B |
| Test videos | `data/test_videos_unseen/` | Tracks C, D, E |
| Physics tests | `tests/test_physics.py` | All tracks |
| Broad project docs | `shared/docs/` | Reference for everyone |

## Rules
1. **Do NOT edit files owned by another track** without coordinating first
2. **Always measure before and after** using Track E's eval pipeline
3. **Save your reports** to your track's `reports/` folder with date stamps
4. **Update MODEL_REGISTRY.md** (in `shared/docs/`) when creating new checkpoints
