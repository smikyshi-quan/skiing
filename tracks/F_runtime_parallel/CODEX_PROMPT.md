# Codex Prompt for Track F — Runtime Parallelization

Paste everything below into the Codex thread that has access to `tracks/F_runtime_parallel/`.

---

Make the alpine ski racing video analysis pipeline run faster through batching, parallelism, and async I/O. This is a pure optimization track — correctness must not change.

**Important: Only do Step 1 (profiling) for now.** The pipeline currently produces incorrect results (speeds 3000+ km/h, G-forces 1400+G). Other people are fixing the correctness issues first. Full optimization work should wait until the pipeline produces physically plausible results. But profiling can and should be done now to understand where time is spent.

## Step 1: Profile (do this now)

Add timing instrumentation to `ski_racing/pipeline.py`. Wrap each major phase in the `process_video()` method with `time.time()`:

- Gate detection (initial frame search)
- Gate tracking (full video pass)
- Perspective transform calculation
- Camera motion compensation
- Skier tracking (full video pass)
- Kalman smoothing
- 3D trajectory transform
- Physics validation

Run the 3 regression videos (`tracks/E_evaluation_ci/regression_videos/` — videos 2907, 2909, 2911) and record per-phase timing for each. Save to `tracks/F_runtime_parallel/reports/performance_profile_YYYYMMDD.md`.

## Future optimization targets (after pipeline correctness is fixed)

1. **Batch GPU inference** — both gate detection and skier tracking currently process one frame at a time. YOLOv8 supports batch inference. Implement batch_size=8 default.

2. **Single video pass** — currently reads every frame twice (once for gates, once for skier). Merge into one pass.

3. **Multi-video parallelism** — when processing a directory, each video is independent. Add `--workers N` flag to `scripts/process_video.py`.

4. **Async video decoding** — OpenCV read is synchronous. Use a threaded producer-consumer pattern for frame decoding.

## Files to read first

1. `tracks/F_runtime_parallel/Track_F_Runtime_Parallelization.docx` — full spec
2. `ski_racing/pipeline.py` — understand the phase structure of `process_video()`
3. `ski_racing/detection.py` — `GateDetector` class
4. `ski_racing/tracking.py` — `SkierTracker` class

## Files you own

- `ski_racing/pipeline.py` (add timing, later batch/parallel)
- `ski_racing/detection.py` (later: batch inference)
- `ski_racing/tracking.py` (later: batch inference)
- `scripts/process_video.py` (later: --workers, --batch-size flags)

## Critical rule

Every optimization must produce IDENTICAL outputs to the unoptimized version. If results change by even one number, the optimization is buggy. Verify by comparing output JSONs before and after.

## How to run

```bash
python scripts/process_video.py tracks/E_evaluation_ci/regression_videos/<VIDEO>.mp4 \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --summary
```

## Deliverable (for now)

- `tracks/F_runtime_parallel/reports/performance_profile_YYYYMMDD.md` with per-phase timing for all 3 videos
