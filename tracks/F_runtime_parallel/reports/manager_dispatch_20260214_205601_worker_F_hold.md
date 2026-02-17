# Manager Dispatch: Worker F HOLD

Timestamp: 2026-02-14 20:56:01
Status: HOLD
Reason: Track F optimization starts last, only after B/C/D produce stable and physically plausible outputs.

## Allowed While Held
- Profiling only (Step 1): produce `tracks/F_runtime_parallel/reports/performance_profile.md`.

## Blocked Until Release
- Batch inference changes
- Single-pass refactor
- Multi-video parallelism
- Async decode changes

## Release Conditions
- B/C/D outputs are stable and meet core correctness direction:
  - Gate F1 remains above threshold target,
  - Auto-cal correction is brought under control,
  - Tracking jump/coverage diagnostics are within target bands,
  - Physics metrics move toward plausibility (speed < 100 km/h, G < 6 on regression).
- Eval pipeline is available for before/after identical-output verification.
