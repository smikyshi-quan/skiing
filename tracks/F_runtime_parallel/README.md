# Track F: Runtime Parallelization

## Owner
Assigned to: _______________

## Goal
Reduce pipeline execution time through batching, parallelism, and async I/O.

**START THIS TRACK LAST.** Only after Tracks B–E are producing stable, correct results.

## Your Spec Document
**Read this first:** `Track_F_Runtime_Parallelization.docx` (in this folder)

## Files You Own (edit these)
| File | What to change |
|------|---------------|
| `ski_racing/pipeline.py` | Batch processing, single-pass architecture |
| `ski_racing/detection.py` | Batch GPU inference for GateDetector |
| `ski_racing/tracking.py` | Batch GPU inference for SkierTracker |
| `scripts/process_video.py` | Add --workers and --batch-size flags |

## Key Rule
**Correctness first.** Every optimization must produce identical outputs to the unoptimized version. Verify by running Track E's evaluation pipeline before and after each change.

## First Step: Profile
Before optimizing anything, add timing instrumentation to pipeline.py and measure where time is actually spent. Save the profiling report to `tracks/F_runtime_parallel/reports/`.

## Dependencies
- **Depends on all other tracks** — don't optimize incorrect code
- Exception: profiling (step 1) can be done early
