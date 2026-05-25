# SkiCoach AI — Metrics & Coaching Pipeline Improvement Plan

## Context

The current pipeline (`technique-analysis/src/technique_analysis/`) produces unstable turn counts, false-positive critical movement-quality labels and downstream coaching complaints (especially "Critical — Knee Angle > 165°"), and metrics whose 3D-derived quantities carry monocular depth noise. Multiple reviewers, market research, and the alpine biomechanics / motion-capture literature converge on the same priority: make the detected skeleton, turn boundaries, and derived metrics accurate enough for an LLM judge to trust.

The plan has eight work blocks: Phase −1, Phase 0, and Phases 1–6. Phase −1 is a hard prerequisite. Phase 0 must land before any algorithmic change. Phases 1–6 are sequenced but each ships independently within its phase.

**Market-research implications baked into this plan**
- Demand exists among motivated improvers who already post ski videos for critique, but they are skeptical of generic AI coaching. The product must show evidence quality and concrete measurements, not broad advice.
- The market is crowded and subscription-sensitive. Public launch, marketing polish, and deploy status are deliberately de-prioritized until the private product loop proves that the analysis itself is useful.
- User-uploaded footage is often imperfect. Low-quality videos should not hide the score; the UI should show the score with a clear warning that poor skeleton detection can make the result misleading, plus concrete filming suggestions.

---

## Cross-language contract changes (applies to every phase below)

Multiple phases add fields to Python dataclasses in `common.contracts.models`. Every active public UI-facing change must land in `MVP/web/lib/analysis-summary.ts` **and** the relevant dashboard/render consumer in the same PR, or the web recap will parse the field but never display it. Field-add-only migration PRs are allowed when they explicitly preserve old behavior and include the follow-up dashboard migration step. This is not a Phase 6 concern — Phases 1 and 2 already add public fields.

Contract fields split into two tiers: **public** fields rendered by the UI dashboard, and **internal diagnostic** fields used only by the harness and debugging. The split is enforced by a serialization mechanism, not by hope.

**Enforcement mechanism (required — current code does not implement this)**

The current serializer at `common/contracts/models.py:10-20` is `_jsonable()`, which recursively `asdict()`s every field. There is no way to add a field that *isn't* serialized into `summary.json`. Before any Tier 2 field is added, the contract must change:

```python
# In TechniqueRunSummary (and any other top-level summary dataclass):
@dataclass(slots=True)
class TechniqueRunSummary:
    # ... existing public fields ...
    diagnostics: DiagnosticsBundle = field(default_factory=DiagnosticsBundle)


@dataclass(slots=True)
class DiagnosticsBundle:
    """All Tier 2 fields live here. Serialized as a single nested object."""
    nyquist_violation: list[str] = field(default_factory=list)
    filter_cutoffs_applied: dict[str, float] = field(default_factory=dict)
    per_segment_low_confidence_fraction: list[float] = field(default_factory=list)
    stage_2_refinement_counts: dict[str, int] = field(default_factory=dict)
    rejected_candidates: list[dict] = field(default_factory=list)
    # ... etc
```

As part of this contract change, the TS interface should declare `diagnostics?: unknown` (opaque, optional). The dashboard never reads it. `mvp_summary.json` (`MVP/run.py:72-86`) is updated to either include a `diagnostics` block or — preferred — to omit it entirely, since the worker doesn't need it.

This nested-object pattern is the cleanest enforcement: a Tier 2 field can only end up in the public dashboard if someone *explicitly* moves it out of `DiagnosticsBundle`, which is a visible code-review event.

**Tier 1 — Public contract (UI dashboard reads these now or after an explicit migration step; must be TS-mirrored)**:

| Phase | Field | Owner type |
|---|---|---|
| 2 | `score_reliability: 'reliable' \| 'limited' \| 'insufficient'` | `QualityReport` |
| 2 | `score_counts_for_progress: bool` | `QualityReport` |
| 2 | `quality_warnings: list[str]` | `QualityReport` |
| 2 | `stance_measurable`, `stance_visibility_fraction` | `QualityReport` |
| 2 | `avg_upper_body_separation` *(new; new formula)* | `TurnSummary` |
| 2 | `avg_com_shift_lateral` *(new; `\|com_shift_x\|`)* | `TurnSummary` |
| 4 | `phases`, `phase_metrics` | `TurnSummary` |
| 5 | `wedge_likely` | `QualityReport` |
| 6 | LLM judge artifact using existing AI-coaching output shape where possible | `TechniqueRunSummary` / worker artifact |

**Tier 2 — Internal diagnostic (lives on `TechniqueRunSummary.diagnostics`)**:

| Phase | Field | Purpose |
|---|---|---|
| 1 | `timestamp_source`, `timestamp_warning`, `analysis_fs`, `end_timestamp_s` | Debug only — user has no use for these |
| 3 | `nyquist_violation: list[str]` | Phase 3 filter audit |
| 3 | `filter_cutoffs_applied: dict[str, float]` | Harness regression on Nyquist downgrade |
| 3 | `per_segment_low_confidence_fraction: list[float]` | Audit short-segment skips |
| 5 | `event_mapping: 'A' \| 'B' \| 'C'` | Tracks the Phase 5 harness choice; not user-facing |
| 5 | `boundary_reliability_by_turn: dict[int, 'high' \| 'medium' \| 'low']` (keyed by `turn_idx`) | Per-turn boundary reliability. Lives entirely inside `DiagnosticsBundle` — adding it to `TurnSummary` would make it public via `_jsonable()`. Feeds run-level score reliability and the metrics-only LLM judge input. |
| 5 | `stage_2_refinement_counts: dict[str, int]` | Detect refinement-rate regressions |
| 5 | `rejected_candidates: list[Candidate]` (with reason) | Harness audits rejection patterns |

**About the TS sync claim**: previous revisions said "the web recap silently drops fields if TS isn't updated." That's not accurate — `server-job-data.ts:271` does a runtime `JSON.parse` and TS type assertions are erase-at-runtime, so unknown fields survive parsing. The real failure mode is that the dashboard (`MVP/web/lib/analysis-summary.ts:400`+) reads *specific* fields by name; a new field that isn't read by `buildTechniqueDashboard()` simply doesn't appear in the UI. The check the harness needs is therefore: "for every Tier 1 field added, is there a corresponding consumer in `buildTechniqueDashboard()` or sibling render code?" Not "did the TS interface get updated."

**Mechanical enforcement (concrete)**
- PR template asks three questions for any Tier 1 addition: (a) is the TS interface updated, (b) is the dashboard render code updated to consume it when the field becomes active, (c) if this is a field-add-only migration PR, is the follow-up consumer migration explicitly linked.
- Phase 0 harness adds a structural diff check between Python `TechniqueRunSummary` (excluding `diagnostics`) and TS `TechniqueRunSummary`. Mismatch is a warning, not an auto-fail (some skews are intentional during phased rollouts).

**`DRILLS` quadruplication** (`lmstudio_coaching.py`, `claude_coaching.py`, `gemini_coaching.py`, `MVP/web/lib/drills.ts`) is a known wart, out of scope for this work, but any drill-field change needs all four updated.

---

## Phase −1 — Dependency manifest (hard prerequisite)

The repo currently has **no Python dependency manifest** (no `requirements.txt`, no `pyproject.toml`, no `Pipfile`). Phases 3 and 5 add SciPy as a hard dependency; the regression harness's tolerance bands are only meaningful under a pinned MediaPipe version. Both require a manifest to exist.

**Deliverables — chosen path (no alternatives)**

Python code spans `MVP/` and `technique-analysis/`, so the manifest lives at the repo root, not under `MVP/`. The chosen path is **root `pyproject.toml` + `uv.lock`** managed by `uv`. Reasons: `uv` is fast, deterministic across platforms, explicitly tracks the full resolution graph, and treats the lockfile as the source of truth rather than a generated artifact.

Top-level dependencies declared in `pyproject.toml`:
- Required: `opencv-python`, `mediapipe`, `numpy`, `ultralytics`, `supabase`, `python-dotenv`, `boto3`, `requests`, `scipy` *(new)*
- Optional: `torch` *(used only for Apple MPS backend selection in `person_detector.py:166`; install path documented but not required for CPU/CUDA paths)*

`uv.lock` is committed to the repo and is what Phase 0's tolerance assertions are pinned *against*. Updating any dependency requires re-running the harness and signing off on any snapshot diffs — same workflow as a Phase 3 or Phase 5 change.

**Why top-level pins alone are not enough**: `ultralytics` pulls in `torch`, `torchvision`, and several other libraries whose minor versions affect detection output. The 5 mm landmark tolerance in Phase 0 is meaningless without transitive determinism, which only a lockfile provides.

**Not acceptable**: a `requirements.txt` with mixed pinned/unpinned entries, or any setup where transitive resolution happens at install time. That's what tolerances silently drift against.

Document the `uv sync`-based install path in `CLAUDE.md` / `AGENTS.md`.

This is its own phase because every downstream phase implicitly depends on it.

---

## Phase 0 — Regression harness

Nothing else is falsifiable without this.

**Fixtures (location matters)**
- All fixtures live in `technique-analysis/tests/regression/fixtures/`. Never under `technique-analysis/artifacts/` — that path is gitignored and reserved for generated outputs.
- Setup script reads from `artifacts/runs/<id>/`, sanitizes (strip absolute paths, remove large binaries), writes the reduced fixture into `tests/regression/fixtures/`.

**Fixture matrix** — explicitly sized so the harness can pin every calibration table entry (rows 1–19). 5–10 clips is not enough; the matrix below sets a floor of ~20–25 distinct fixtures, organized by what they exercise:

| Fixture group | Min count | Purpose | Calibration rows exercised |
|---|---|---|---|
| Carved, side-on, CFR, good light | 3 | Best-case baseline, snapshot pinning | 1, 3, 5, 7 |
| Carved, oblique angle | 2 | Camera-angle degradation | viewpoint pipeline |
| Carved, follow-cam | 2 | Stance-measurable false / score-warning behavior | 10 |
| Wedge / snowplow, hand-labeled | 3 | `wedge_likely` detector calibration; turn-count recall on degraded signal | 17, 18 |
| VFR clips, varied phone backends (iOS, Android, action-cam) | 3–4 | PTS full-stream validation correctness across decoder backends; fallback path coverage | 11 |
| Low-fps clips (10–15 fps, intentionally subsampled) | 2 | Nyquist guard exercises both downgrade paths | 15, 16 |
| Multi-skier scene (other person briefly in frame) | 2 | Cross-skier guard (Phase 3); short-segment rejection | 4, 13 |
| Scene cuts within one clip | 2 | Smoother-reset path; state-machine reset; PTS handling across cuts | 11, 13 |
| Tracker dropout segments (≥ 0.5 s low-confidence) | 2 | Gap-fill behavior; interpolate-then-filter sequence | 2, 5, 13 |
| Long sweeping turns (> 5 s by hand label) | 2 | Under-segmentation flag; long-turn flag threshold | 6 |
| Short carved turns (≤ 1 s by hand label) | 2 | `find_peaks` distance floor; over-segmentation rejection | 3, 12 |
| Hand-labeled switches per Phase 5 mapping | 3+ (overlap OK) | Mapping A/B/C comparison against ground truth | 14, 19 |

**Hand-labeling protocol**: for each "hand-labeled" clip, the label includes turn count, turn-direction polarity (left/right per turn), switch timestamps to the nearest 100 ms, and a free-form notes field. Two labelers per clip; disagreements adjudicated by a third or excluded. Labels live in `tests/regression/fixtures/labels/<clip>.json`.

**Snapshot fixtures** (separate from the matrix above): the existing run outputs pinned for tolerance-band assertions. Same provenance rules — sanitized from `artifacts/`, copied into `tests/regression/fixtures/`, never committed under `artifacts/`.

**Harness execution model — in-process, not file-roundtripped**

The current pipeline writes structured `summary.json` and `metrics.csv` outputs, and may also render `overlay.mp4`. Frame-level landmark trajectories and the `DiagnosticsBundle` are not exposed as separate structured files. Rather than add new artifact file formats, the harness invokes the pipeline **in-process**:

```python
from technique_analysis.free_ski.pipeline.orchestrator import TechniqueAnalysisRunner

def harness_run(clip_path, config):
    runner = TechniqueAnalysisRunner(config=config)
    summary = runner.run(clip_path)   # returns TechniqueRunSummary directly
    return summary   # full object including .diagnostics and .turns[i].phase_metrics
```

This avoids inventing a new debug-trajectory artifact format. The harness inspects the same Python objects the worker (`MVP/worker.py`) inspects, which means the harness is testing the actual API surface rather than a file-serialized approximation. Per-landmark trajectories that the harness wants to assert against can be exposed by adding an opt-in field on `TechniqueRunConfig` (e.g., `keep_landmark_trajectories: bool = False`) that the orchestrator honors — defaulting to `False` so production runs don't pay the memory cost, set to `True` by the harness.

If a future need arises for off-the-record fixture inspection (e.g., a reviewer who isn't running Python), the harness can additionally write a `debug_trajectories.npz` per clip on opt-in — but this is a Phase 0 enhancement, not a baseline requirement.

**Two assertion modes — required to keep the plan shippable through Phases 3 and 5:**
- **Tolerance-band assertions** (auto-fail): turn count ±1, score ±5, `low_confidence_fraction` ±5%, warnings set equality, and ankle-visibility / `stance_measurable` classification equality once Phase 2 adds the structured fields.
- **Snapshot diffs** (human review, never auto-fail): frame-level trajectories. Phase 3 and Phase 5 will deliberately move these — the harness surfaces the diff, a human signs off, and the new snapshot is re-pinned.

**Landmark tolerance**
- ~5 mm per landmark, looser on distal joints.
- **Explicitly documented as regression tolerance under pinned MediaPipe version + backend, not correctness tolerance.** MediaPipe monocular RMSE ~56 mm sets the upper bound on cross-version validity.

**Stance-measurable validation clips**
- Same skier filmed from side-on, follow-cam, oblique angles.
- Current `detect_viewpoint()` returns a warning string or `None`; it does not yet return structured labels. Phase 0 pins the underlying ankle-visibility behavior, and Phase 2 wraps that same heuristic into structured `stance_measurable` / `stance_visibility_fraction` fields.

**Sampling rate**
- Derive analysis `fs` from PTS where available (see Phase 1), not from container metadata.

---

## Common gap-handling module (shared by Phases 1, 2, and 3)

Three phases all answer the same question: "what do we do with masked / missing samples in a time series before downstream code consumes it?" — the segmenter mask fix (Phase 1), pose gap-filling (Phase 2), and the interpolate-then-filter sequence (Phase 3). To avoid three subtly-different implementations, the plan adds one module:

```
technique_analysis/common/gap_handling.py

interpolate_short_gaps(series, mask, max_gap_frames, kind='linear') -> series
    # Linear interpolation across contiguous masked runs ≤ max_gap_frames.
    # Edge gaps (start/end of segment) handled by `decayed_edge_fill` separately.

decayed_edge_fill(series, mask, edge) -> series
    # For runs touching the start or end of a segment, carry the nearest
    # valid value with a confidence decay; mark filled frames as low-confidence.

run_gap_mask(mask, max_gap_frames) -> mask_of_acceptable_gaps
    # Identifies which masked runs are short enough to interpolate vs.
    # long enough to leave as missing.

reapply_mask(series, original_mask) -> series_with_mask
    # After upstream consumers (filters, scoring) have used the interpolated
    # series, this re-marks the interpolated regions as low-confidence so
    # downstream code doesn't over-trust them.
```

Phases use it as follows:
- **Phase 1 segmenter fix**: replaces `0.0`-masking at `segmenter.py:117-120` with a single call to `interpolate_short_gaps(series, low_conf_mask, max_gap_frames=N)` on the lateral-CoM trajectory before zero-crossing/extrema detection. `reapply_mask` is then called on the interpolated signal's reliability metadata so any turn whose boundary lands inside an interpolated region is flagged low-confidence downstream — the segmenter's actual output (turns) is unchanged shape; what changes is the per-frame mask that travels alongside it.
- **Phase 2 gap filling**: `_fill_pose_gaps` at `orchestrator.py:91-113` is rewritten as `interpolate_short_gaps + decayed_edge_fill` per landmark coordinate.
- **Phase 3 interpolate-then-filter**: same module functions, called per landmark/coordinate/metric trajectory immediately before `sosfiltfilt`.

Acceptance criteria for the module (Phase 0 harness):
- Identical output for the three current call sites (i.e., this is a refactor, not a behavior change in isolation — behavior changes happen in the phases that consume it).
- `max_gap_frames` configurable per call site; default 10 frames as in Phase 2.
- Edge-of-segment behavior matches the current decayed carry-forward path at `orchestrator.py:99-112`.

---

## Phase 1 — Local correctness bugs (low-risk, independently shippable)

**Segmenter mask fix** at `segmenter.py:117-120`
Replace 0.0-masking with `interpolate_short_gaps` from the common gap-handling module. This is the Phase 3 interpolate-then-filter sequence's first step lifted forward — not a separate algorithm.

**Smoother reset**
Add `LandmarkSmoother.reset()`; call it on scene cuts and tracking-segment boundaries. Keep explicit even after Phase 3 makes it architecturally redundant — streaming code path will still exist.

**PTS-aware frame iterator** (`video_io.py:77-104`)
Currently `timestamp_s = frame_idx / native_fps` at line 100. Variable-frame-rate phone footage is silently treated as constant-rate.

Two separate concerns, easy to conflate:
1. **Is PTS trustworthy?** Decided by **full-stream validation** in pass 1 — every frame's PTS is checked for monotonicity before pass 2 (pose extraction) begins.
2. **What is `analysis_fs`?** Computed from the actual analyzed-frame timestamps *after* subsampling — not from the validation pass.

The orchestrator (`orchestrator.py:289-299`) reads every frame with `max_fps=None` for ByteTrack and accumulates `frame_timestamps` only for the subsampled cadence (`resolved_max_fps`). A 60 fps clip analyzed at 20 fps must filter as if sampled at 20 Hz, not 60 Hz — so `analysis_fs` is derived from `frame_timestamps`, never from the validation pass's native-rate samples.

A first-N-frames probe alone would be insufficient — PTS can pass the first 30 frames cleanly and then reset or stall mid-stream (observed on some Android encodes and on clips with embedded chapter markers). An earlier design tried to carry both PTS and frame_idx timestamps through iteration and commit at end-of-stream, but that conflicts with the streaming pipeline: the orchestrator must select analysis frames and call `extractor.extract(frame, timestamp_s)` *during* iteration, and `timestamp_s` has to be one value. If we later fall back, `FramePose.timestamp_s` is already wrong on every stored pose. Full-stream validation in pass 1 avoids both failure modes.

**Two-pass design — pass 1 validates, pass 2 runs the pipeline against the committed clock.**

```python
def _validate_pts_full_stream(video_path, native_fps):
    """Pass 1: full-stream validation. Uses cap.read() so pass 1 and pass 2
    observe the same decoded frame count; decoded pixels are discarded.

    Returns (committed_source, pts_series, end_timestamp_s, warning).
    On 'pts': pts_series is the full per-frame timestamp array, end_timestamp_s
              is the end-exclusive video duration (pts_series[-1] + median_dt).
    On 'frame_idx': pts_series and end_timestamp_s are None.
    """
    cap = cv2.VideoCapture(str(video_path))
    pts_series = []
    while True:
        ok, _frame = cap.read()   # read() — count must match pass 2 exactly; discard pixels
        if not ok:
            break
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pts_s = pts_ms / 1000.0
        # First frame at 0.0 is normal — V4L and some FFmpeg backends return 0 for FirstCapture.
        # Require strict monotonic increase from frame 1 onward, and a positive final timestamp.
        if pts_series:
            if pts_s <= pts_series[-1]:
                cap.release()
                return "frame_idx", None, None, "pts_not_monotonic_mid_stream"
        else:
            if pts_s < 0:
                cap.release()
                return "frame_idx", None, None, "pts_negative_at_start"
        pts_series.append(pts_s)
    cap.release()

    if len(pts_series) < 5:
        return "frame_idx", None, None, "video_too_short_to_validate_pts"
    if pts_series[-1] <= 0:
        return "frame_idx", None, None, "pts_never_advanced_from_zero"

    intervals = [b - a for a, b in zip(pts_series, pts_series[1:])]
    median_dt = sorted(intervals)[len(intervals) // 2]
    probe_fs = 1.0 / median_dt if median_dt > 0 else 0.0
    if not (0.25 * native_fps <= probe_fs <= 4.0 * native_fps):
        return "frame_idx", None, None, "pts_fs_implausible"

    # End-exclusive duration: pts_series[-1] is the *start* of the last frame,
    # not the end of the video. _build_segments uses `< end_s` (orchestrator.py:205),
    # so without this offset the final frame is silently dropped from the last
    # tracking segment.
    end_timestamp_s = pts_series[-1] + median_dt
    return "pts", pts_series, end_timestamp_s, None


def iter_frames(video_path, native_fps, committed_source, pts_series):
    """Pass 2: actual iteration with a single, committed timestamp source."""
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if committed_source == "pts":
            timestamp_s = pts_series[frame_idx]
        else:
            timestamp_s = frame_idx / native_fps
        yield frame_idx, timestamp_s, frame
        frame_idx += 1
    cap.release()
```

The orchestrator calls `_validate_pts_full_stream` *before* anything in the analysis pipeline runs:

```python
committed_source, pts_series, end_timestamp_s, warning = _validate_pts_full_stream(
    video_path, native_fps
)
# All timestamp-related fields are diagnostic. Per the Tier 1/Tier 2 contract,
# they live on summary.diagnostics, never on QualityReport.
diagnostics.timestamp_source = committed_source
diagnostics.timestamp_warning = warning
diagnostics.end_timestamp_s = end_timestamp_s

# Propagate duration to VideoMetadata BEFORE downstream consumers use it.
# Default at video_io.py:54 is `frame_count / native_fps` — wrong on VFR clips.
# Use end_timestamp_s (= last_pts + median_dt), not pts_series[-1]: _build_segments
# at orchestrator.py:205 uses `< end_s`, so without the offset the final frame is
# silently dropped from the last tracking segment.
if committed_source == "pts":
    metadata = dataclasses.replace(metadata, duration_s=end_timestamp_s)

# Now iterate; pose extraction sees correct timestamps from the first frame.
# Pass 1 used read() to build pts_series, so pass 2's read() count matches by construction.
for frame_idx, timestamp_s, frame in iter_frames(
    video_path, native_fps, committed_source, pts_series
):
    # ... pose extraction ...
```

**Per-backend behavior is tracked in Phase 0.** The VFR fixture group (3–4 clips across iOS / Android / action-cam encodes) exercises every backend the worker is likely to see. PTS validity per backend is recorded in `tests/regression/fixtures/backend_notes.md`. Since both passes use identical `read()` calls, there's no `grab()`/`read()` divergence to catalog.

**Why end-exclusive duration matters**: `_build_segments(..., metadata.duration_s)` filters frames with `start_s <= m.timestamp_s < end_s`. On a 60 s VFR clip whose last PTS is `59.967 s` (median_dt = `0.033 s`), setting `duration_s = 59.967` excludes the final frame; setting `duration_s = 60.0` (the end-exclusive boundary) includes it. The CFR default at `video_io.py:54` is `frame_count / native_fps` — wrong by both the VFR rate drift AND the end-exclusive offset. PTS + median_dt fixes both with one calculation.

**Pass 1 uses `read()`, not `grab()` — for count consistency with pass 2**

Earlier revisions of this plan proposed pass 1 with `cap.grab()` for speed, then catching count mismatches in pass 2. That design was unsafe: pass 2 already does pose extraction, so a mid-iteration mismatch raise would throw away minutes of work. The clean alternative is to use `read()` in both passes — same call, same decode tolerance, identical frame counts by construction.

Costs and tradeoffs:
- Pass 1 with `read()` is a full decode pass that discards pixels. On a 60 s 30 fps clip: ~1–3 s on the MacBook worker. Acceptable.
- Count consistency is guaranteed because both passes use the identical decoder call. No backend-dependent `grab()` vs. `read()` divergence to track.
- Memory cost of `pts_series` is ~8 bytes per frame, ~14 KB per minute of 30 fps video. Negligible.
- Pass 1 is required even on clips that ultimately fall back to `frame_idx`, because the fallback decision must be made before iteration starts.

Never seek — `CAP_PROP_POS_FRAMES` is unreliable on many OpenCV builds; both passes start from the beginning by re-opening the capture.

**No mid-pipeline raises**: every PTS / count failure mode is detected in pass 1 and converted to a clean `frame_idx` fallback before pass 2 starts. Pass 2 is a pure consumer of an already-committed clock; it cannot encounter PTS surprises.

`analysis_fs` is computed in the orchestrator after `frame_timestamps` is fully assembled, from the median inter-sample interval of the *analyzed* frames:

```python
def _derive_analysis_fs(frame_timestamps, native_fps, resolved_max_fps):
    if len(frame_timestamps) < 2:
        return resolved_max_fps or native_fps
    intervals = [b[1] - a[1] for a, b
                 in zip(frame_timestamps, frame_timestamps[1:])
                 if b[1] > a[1]]
    if not intervals:
        return resolved_max_fps or native_fps
    median_dt = sorted(intervals)[len(intervals) // 2]
    return 1.0 / median_dt if median_dt > 0 else (resolved_max_fps or native_fps)
```

**`DiagnosticsBundle` additions (Tier 2 — never on `QualityReport`)**
- `timestamp_source: 'pts' | 'frame_idx'` — decoder-level decision
- `timestamp_warning: str | None` — only populated when the irregularity is large enough to affect turn boundaries; not on every fallback
- `analysis_fs: float` — derived from `frame_timestamps`, used by every Phase 3 / Phase 5 filter
- `end_timestamp_s: float` — end-exclusive video duration (`pts_series[-1] + median_dt`); used to patch `VideoMetadata.duration_s`

Fallback to `frame_idx` is recorded for debugging and regression but **not surfaced as a user-facing quality warning**. None of these fields are public — see the global contract table at the top of the document for the Tier 1/Tier 2 split.

**Other Phase 1 items**
- `compute_upper_body_quietness` → `dataclasses.replace` instead of manual reconstruction.
- Centralize the seven duplicated confidence thresholds into `TechniqueRunConfig`.
- `compute_jitter_score` at `smoother.py:81` → divide by `dt` so it's FPS-invariant. Audit every other "per-frame displacement" metric in the same pass.

---

## Phase 2 — Cheap wins (each independently shippable)

**Gap filling** (`_fill_pose_gaps` at `orchestrator.py:91-113`)
- Rewrite using `interpolate_short_gaps` + `decayed_edge_fill` from the common gap-handling module.
- `max_gap_frames = 10` (gap-fill max length, calibration row 5). Gaps longer than this are left masked.
- Never interpolate across scene cuts or tracking-segment boundaries — the module's `run_gap_mask` is computed per segment, not across the whole run.

**Skeletal refiner** (`skeletal_refiner.py:64-69`)
- Disable knee/ankle leg-chain refinement by default. Let Phase 1/Phase 2 gap-filling handle missing legs. Keep arm refinement.
- This removes the bias source for the "Critical — Knee Angle > 165°" movement-quality label (occluded knees were being placed on the straight hip-ankle line).

**Upper-body quietness — RENAME, do not silently change semantics**

`turn.avg_upper_body_quietness` is consumed by the web dashboard at `MVP/web/lib/analysis-summary.ts:410`. Changing its formula in place silently invalidates the dashboard's score thresholds.

Migration (strict — no semantic overwrite at any point):
1. **PR A — add new field, leave old field alone**: introduce `turn.avg_upper_body_separation` computed by the new formula (shoulder-line vs. hip-line angular separation variance). `turn.avg_upper_body_quietness` keeps its **old formula** (`nose.x` variance). Both fields ship together. Dashboard still reads the old field; behavior unchanged.
2. **PR B — dashboard migration**: `MVP/web/lib/analysis-summary.ts` switches to reading `avg_upper_body_separation`. Calibrate `positiveScore(value, floor, ceiling)` floor/ceiling values against the new formula's distribution on the harness fixture set. `avg_upper_body_quietness` is no longer consumed but is still computed (still using the old formula).
3. **PR C — removal**: delete `avg_upper_body_quietness` from `TurnSummary` and the Python computation. TS interface drops the field. One release window between PR B and PR C minimum, so any other consumer (notebooks, exports, analytics) surfaces before the field disappears.

The old field carrying the new formula's value is explicitly forbidden — it would mean both names mean the new thing, and any analytics or notebook pinned to the old field's old-formula expectations would silently break.

Raw `nose.x` variance is never the answer long-term — it conflates camera pan and skier traverse with body instability. The old field keeps it only until removal; nothing new consumes it.

**Z-axis downgrade — RENAME, no semantic overwrite**

`turn.avg_com_shift_3d` is consumed by the dashboard at `analysis-summary.ts:411`. Same three-PR migration:
1. **PR A**: introduce `turn.avg_com_shift_lateral` (= `|com_shift_x|`, no z component). `turn.avg_com_shift_3d` keeps the **old** `√(x² + z²)` formula.
2. **PR B**: dashboard switches to `avg_com_shift_lateral`; recalibrate `positiveScore` thresholds in the same PR.
3. **PR C**: remove `avg_com_shift_3d` after the migration window.

Notes:
- The segmenter's direction signal at `segmenter.py:154` uses **signed** `com_shift_x` (internal to the segmenter; not a renamed field — it never had a public name).
- Other `√(x² + z²)` constructions inside Python become `DiagnosticsBundle` entries rather than top-level summary fields. Anywhere the dashboard currently reads them follows the same three-PR migration.

**Why three PRs not one**: silent semantic changes corrupt every consumer with pinned thresholds, ranges, or test fixtures. The dashboard's `positiveScore(value, floor, ceiling)` is exactly this case — its floor/ceiling were tuned against `√(x²+z²)`, not `|x|`, and the same numeric `0.05` means different things between formulas. Separating "add new field" from "switch consumer" from "remove old field" gives every consumer a chance to migrate explicitly and leaves a working rollback at every step.

**Metric and score reliability gating**
- Compute `low_confidence_fraction` over the same windows that feed turn metrics and the final score (start τ = 0.3; calibrate in Phase 0).
- Set `QualityReport.score_reliability ∈ {reliable, limited, insufficient}` from pose confidence, stance visibility, segment length, boundary reliability, and wedge / follow-cam degradation.
- Always keep the numeric score visible. When `score_reliability = insufficient`, set `score_counts_for_progress = false` and add a user-facing warning that the video quality is too low to support reliable skeleton detection, so the result may be misleading.
- The warning should be direct and actionable: use a better angle, higher resolution, closer camera, steadier shot, and cleaner single-skier framing.
- Feed the reliability fields into the metrics-only LLM judge as input context. Do not create or suppress deterministic rule tips here.

**Stance-measurability gating** (not "viewpoint classification")
- Promote `viewpoint.py`'s output to first-class fields on `QualityReport`:
  - `stance_measurable: bool`
  - `stance_visibility_fraction: float`
- Gate stance-width metrics and any lateral-stance interpretation on `stance_measurable`.
- Do not hide the score when stance is not measurable. Mark affected metrics unreliable, keep the score visible, and explain why the result may be misleading.
- **Do not** claim multi-class viewpoint classification (side-on / follow-cam / oblique). That's a separate research task. The current heuristic is binary: are ankles visible enough to make stance measurable, or not.

---

## Phase 3 — Zero-phase smoothing, two filters, two cutoffs

Restructure so smoothing happens **after** poses are assembled, **per tracking segment**.

**Two filter targets**
- **Landmark smoothing**: target effective 4th-order Butterworth zero-phase response. Implementation:
  ```python
  sos = scipy.signal.butter(N=2, Wn=cutoff_hz / (analysis_fs / 2), btype='low', output='sos')
  smoothed = scipy.signal.sosfiltfilt(sos, signal)
  ```
  - `N=2` prototype + `sosfiltfilt` → effective 4th-order (matches Winter / Frontiers convention).
  - `output='sos'` + `sosfiltfilt` is SciPy's recommended path for numerical stability.
  - `cutoff_hz` starts at 6 Hz; pin via residual analysis on harness clips (5–8 Hz range).
- **Turn-detector signal**: same construction at **0.5 Hz** (coarse) and **3.0 Hz** (fine), per the two-stage detection in Phase 5. Both use `N=2` prototype + `sosfiltfilt`.

**Pipeline order — no shortcuts**
1. Split into tracking segments / scene-cut-bounded segments.
2. `interpolate_short_gaps` from the common gap-handling module.
3. **Uniform-grid resampling** (see below).
4. `sosfiltfilt` on the uniform grid.
5. Map filtered values back to the original analyzed frame timestamps via linear interpolation.
6. `reapply_mask` so interpolated regions don't get reported as reliable downstream.

**Uniform-grid resampling (required for VFR correctness)**

`sosfiltfilt` operates on an evenly-indexed array — applying it directly over irregularly-spaced timestamps is mathematically wrong (unequal time gaps treated as equal). Per tracking segment, before filtering:

- Compute the segment's uniform grid: `t_uniform = arange(t0, t1, 1/analysis_fs)`.
- Linear-interpolate **every signal that will be filtered** from the segment's actual timestamps onto `t_uniform`. Specifically:
  - **Per landmark, per coordinate** (33 landmarks × 3 coordinates = 99 per-frame signals for MediaPipe), pre-Phase-3 landmark smoothing.
  - **Per derived metric trajectory** (lean, edge, hip_tilt, com_shift_x, ...) used downstream.
  - **The lateral-CoM turn-detector signal** before each of the two segmenter filters (0.5 Hz coarse and 3.0 Hz fine).
- Filter on the uniform grid.
- Map back to the original analyzed frame timestamps via linear interpolation, for downstream consumers that index by frame.

For approximately-uniform clips (CFR phone footage) the per-coordinate resampling is a no-op; for true VFR clips it's the difference between correct and incorrect filtering. The CPU cost is `O(landmarks × coords × frames)` per segment, which is fast — measure but don't pre-optimize.

**Nyquist guard** (required — SciPy will raise if `Wn ≥ 1`)
Nyquist frequency is `analysis_fs / 2`. For each filter (`landmark_cutoff_hz = 6 Hz`, `turn_coarse_hz = 0.5 Hz`, `turn_fine_hz = 3.0 Hz`):
- If `analysis_fs ≥ 2 × cutoff_hz × safety_margin` (safety margin = 1.25): use the nominal cutoff.
- If `cutoff_hz < analysis_fs / 2 < cutoff_hz × 1.25` (cutoff is below Nyquist but close): lower the cutoff to `0.2 × analysis_fs` (= 40% of Nyquist).
- If `analysis_fs ≤ 2 × cutoff_hz`: **skip the filter**, flag the segment `low_confidence`, and append the per-segment flag to `summary.diagnostics.nyquist_violation` (not `QualityReport` — internal-only). At 10 fps a 6 Hz landmark filter is meaningless; refusing is the honest answer.

The 0.5 Hz turn-coarse filter is almost never at risk. The 6 Hz landmark filter and 3 Hz turn-fine filter can both run into the guard on heavily-subsampled or low-fps clips.

**Edge effects on short clips**
- Free-ski clips are often 6–10 s. Per-segment `sosfiltfilt` at effective 4th order needs reasonable length to avoid 14–170% edge error reported in the biomechanics filtering literature.
- For segments **< 3 s (< ~60 samples at 20 fps)**: **skip Butterworth, flag low-confidence**. Savitzky-Golay is a tempting alternative but its short-segment behavior on ski trajectories has not been benchmarked here; introducing it without a Phase 0 test arm is premature. Add SG only after the harness has a comparison column for it.
- For longer segments: `padtype='odd'`, default `padlen` (SciPy computes `padlen ≈ 9` for our N=2 prototype). The short-segment skip rule above prevents the only failure mode (`padlen ≥ len(signal) - 1`).

**Cross-skier guard**
- Reject tracking segments shorter than 1.5 s before they enter the smoothing pipeline. Short re-acquisitions are dominated by tracker noise or another person briefly in frame.

**Hypothesis to verify** (don't bake in)
- That removing the double-smoothing (EMA + rolling mean) recovers a useful smoothness signal at `scoring.py:107`. The harness measures this — don't assume.

---

## Phase 4 — Turn-phase-aware scoring (3-phase)

Decompose each turn into **initiation / steering / completion**. Not 4-phase — 4 phases require edge-change detection that's not reliable from monocular video.

**Phase boundaries** (starting values, calibrated in harness)
- Initiation: 0–25% of turn duration.
- Steering: 25–75%.
- Completion: 75–100%.
- Optionally refine the steering apex from the filtered-signal extremum within the turn (this is a sub-phase landmark, not a boundary).

**Phase-specific scoring**
- Low edge/lean is normal in initiation and completion — don't penalize.
- Peak edge/lean expected near steering apex.
- Balance/CoM scoring only inside steering (fixes the upright-frame penalization at `scoring.py:76-83`).
- The scoring movement-quality label `Critical - Knee Angle` only applies inside steering, not at transitions; any LLM judge input derived from that label inherits the same phase gate. (Skeletal refiner fix in Phase 2 already reduces false-positives; the phase gate is belt-and-suspenders.)

---

## Phase 5 — Segmenter rewrite

**Gating**: Phase 0 (harness), Phase 1 (mask fix, PTS), and Phase 3 (zero-phase filtering) must be in place.

**Important framing**: This phase **borrows the architectural pattern** from the Frontiers gyroscope turn-detection paper — two-stage detection plus a sign-alternation state machine. It does **not implement that algorithm** on the SkiCoach signal. Their input is boot-mounted gyroscope roll-rate (native ~100 Hz, mechanically coupled to ski edging, single periodic axis). Ours is monocular lateral CoM (~30 fps, projected from 3D body motion onto an arbitrary camera axis).

**The event-mapping problem (must be resolved by the harness before production rewrite)**

In the Frontiers paper, extrema of gyro **roll-rate** correspond to **switch events** (skis are rotating fastest at the switch moment). In SkiCoach, the signal is lateral CoM **position** — and extrema of CoM position correspond to **turn apex** (maximum lateral displacement), not switch. The switch event in a position signal is closer to the **zero-crossing** between consecutive extrema.

There are at least three candidate mappings, and which one matches hand-labeled switches is an empirical question, not an architectural choice:

| Mapping | What it detects | Closest to gyro-paper analog |
|---|---|---|
| **A. Position extrema** | Turn apex (boundary of the steering phase) | No — apex ≠ switch |
| **B. Zero-crossings between extrema** | Switch event by definition | Geometric analog |
| **C. Velocity extrema** (extrema of `d/dt` of position) | Maximum lateral speed; close to switch on smooth turns | Mathematical analog (gyro-rate ≈ d/dt of angle) |

**Phase 5 prerequisite — instrument first, then rewrite:**
- Implement all three mappings against the 0.5 Hz / 3.0 Hz filtered signals.
- Measure each against the golden-clip hand-labeled switch timestamps.
- Pick the production mapping based on harness results, not on the gyro-paper analogy.
- Document the chosen mapping in `summary.diagnostics.event_mapping` so downstream code (and the harness) can be re-evaluated when the mapping changes. Not on `QualityReport` — the user has no use for this.

Stage definitions below assume the harness has picked a mapping. The structure (coarse → fine → state machine) is mapping-independent; the specific event each stage detects depends on the choice.

The harness must measure stage-1, stage-2, and stage-3 outputs against hand-labeled events **separately**, so we can tell which stage is failing when it does.

**Candidate shape (mapping-independent)**

Every stage produces and consumes candidates with the same shape. Fields added by later stages are populated as the candidate flows through the pipeline:

```
Candidate {
    # --- Stage 1 fills these ---
    timestamp: float                # seconds, on the analysis_fs grid recorded in summary.diagnostics
    polarity: -1 | +1               # required by Stage 3 — see per-mapping definition below
    amplitude: float                # signal magnitude associated with the event
    mapping: 'A' | 'B' | 'C'        # source mapping; recorded for harness traceability
    stage_origin: 1                 # set to 1 after Stage 1, may be updated by Stage 2

    # --- Stage 2 fills these (always set, even on no-refinement paths) ---
    stage_2_refinement: 'refined' | 'kept_unrefined' | 'multi_nearest' | 'dropped_polarity_mismatch'
    refined_timestamp: float | None # set when stage_2_refinement == 'refined' or 'multi_nearest'

    # --- Stage 3 / temporal gate fills these ---
    label: 'switch' | 'noise' | 'eliminated' | None  # final classification
    boundary_reliability: 'high' | 'medium' | 'low'  # derived from stage_2_refinement + confidence

    # Dropped candidates (stage_2_refinement == 'dropped_polarity_mismatch'
    # or label == 'eliminated') are retained in summary.diagnostics arrays
    # so the harness can audit rejection rates.
}
```

`boundary_reliability` is derived deterministically per candidate:
- `refined` + clean polarity + in-gate → `high`
- `kept_unrefined` OR `multi_nearest` → `medium`
- Any candidate retained only because the rescue pass re-promoted it → `low`

The aggregate per-turn label (`boundary_reliability_by_turn[turn_idx]`) is the minimum reliability across the turn's two boundary candidates (start and end). This aggregate lives in `summary.diagnostics`, not on `TurnSummary` — putting it on `TurnSummary` would expose it publicly via `_jsonable()`, which contradicts the Tier 1/Tier 2 split.

**Polarity is defined per mapping** (this is the part Stage 3 cannot live without):
- **Mapping A (position extrema)**: `polarity = sign(signal[t])` at the extremum. Positive maxima carry `+1`, negative minima carry `-1`.
- **Mapping B (zero-crossings)**: the signal is zero at the event, so polarity is **transition direction**, not value. `polarity = +1` for negative→positive crossings, `-1` for positive→negative crossings. Computed from `sign(signal[t+ε]) - sign(signal[t-ε])`.
- **Mapping C (velocity extrema)**: `polarity = sign(velocity[t])` at the extremum (where `velocity = d/dt(signal)`). This is the closest analog to the Frontiers paper's gyro-rate extrema, since gyro-rate is mathematically the derivative of orientation.

**Stage 1 — Coarse detection**
- Apply the chosen event-mapping (A, B, or C) to the **0.5 Hz** `sosfiltfilt` signed-lateral-CoM signal.
- For mapping A or C: `find_peaks(signal)` and `find_peaks(-signal)` run separately, both results merged and chronologically sorted. This is the explicit default — do not use `find_peaks(|signal|)`; it discards sign information needed by Stage 3.
- For mapping B: detect sign changes; each zero-crossing is a candidate.
- Parameters (for find_peaks paths):
  - `prominence`: amplitude floor; start from current `_MIN_AMPLITUDE_COM = 0.01`, calibrate in harness.
  - `distance ≈ 0.5 s × analysis_fs`. **Not 1.0 s × fs** — the Frontiers paper reports short carved turns at 1.04 ± 0.41 s, so the lower tail (~0.6 s) gets dropped by a 1.0 s floor. Let `prominence` carry more of the over-segmentation rejection.

**Stage 2 — Fine-tuning**
- For each stage-1 candidate, locate the precise event timestamp by applying the same mapping (A/B/C) to the **3.0 Hz** `sosfiltfilt` signal within an asymmetric window around the candidate (start ±0.3 s, calibrate).
- The window can be wider or shifted relative to the IMU paper's settings; do not assume the IMU paper's window width transfers.

**Stage 2 failure policy** (explicit, not implicit):

| Situation | Action | `stage_2_refinement` flag |
|---|---|---|
| Exactly one matching event in the window | Use refined timestamp, polarity from refined event | `refined` |
| No matching event in the window | Keep Stage 1 candidate timestamp, set candidate's `boundary_reliability = medium` | `kept_unrefined` |
| Multiple matching events in the window | Pick the one **nearest in time** to the Stage 1 candidate (not highest amplitude — nearest preserves the coarse-stage's structural decision) | `multi_nearest` |
| Polarity disagreement between Stage 1 and Stage 2 candidate | Drop the candidate entirely | `dropped_polarity_mismatch` |

**Never expand the window dynamically.** Variable-window detectors are hard to analyze and produce time-varying behavior that the harness can't pin. The window width is a single calibrated parameter (row 9), not a per-candidate decision.

Each Stage 2 outcome is recorded on the `Candidate` (internal) and aggregated on `summary.diagnostics.stage_2_refinement_counts` so the harness can spot regressions in the refinement rate (e.g., a sudden spike in `kept_unrefined` signals a 3.0 Hz signal degradation worth investigating). Not on `QualityReport` — the user has no use for refinement-rate counts.

**Switch-acceptance temporal gate** (from the Frontiers paper)
Before reaching Stage 3, a candidate pair `(c_i, c_{i+1})` with opposite polarity is only labeled `switch` if `0.3 s < (c_{i+1}.timestamp − c_i.timestamp) < 5.0 s`. Pairs outside this band are downgraded to `noise` or `eliminated`. The lower bound rejects intra-turn wiggle that survived `distance`; the upper bound rejects tracker-driven gaps that look like one giant turn. Both bounds are calibration knobs — the Frontiers numbers are starting points, not validated for monocular CoM.

**Stage 3 — Sign-alternation state machine**
- Walk chronologically-sorted stage-2 switches. Maintain `last_accepted_polarity ∈ {positive, negative, none}`.
- **Polarity, not direction**: use internal labels `positive` / `negative` for the signal's sign, not `left` / `right`. With monocular CoM, camera orientation can invert or distort the absolute direction. Left/right is a derived label resolved against the golden clips and the camera-orientation heuristic, not part of the detector's core state.
- For each candidate: if polarity matches `last_accepted_polarity`, classify as `noise` and either discard or merge with the prior switch; otherwise accept and flip `last_accepted_polarity`.
- Optional rescue pass: when two same-polarity extrema appear consecutively, re-promote the larger-prominence candidate (the genuinely correct switch may have been filtered too aggressively upstream).

**State machine reset conditions** (strict alternation is valid for linked turns, not for interrupted clips):
- Scene cuts
- Tracking-segment boundaries
- Long missing intervals (> ~1.5 s of low-confidence frames)
- Explicit low-confidence stretches (`low_confidence_fraction > τ` over a contiguous window)

**Failure modes the plan acknowledges (not all handled)**
- **Over-segmentation** (wiggle in steering): handled by `distance`, `prominence`, and stage 3.
- **Under-segmentation** (very long sweeping turns on steep terrain): not handled by either parameter. Fallback heuristic: if a "single turn" exceeds ~5 s, flag for manual inspection or fall back to hip-tilt secondary segmentation.
- **0.5 Hz signal damping short turns**: the paper's own caveat. The two-stage architecture mitigates but doesn't eliminate; expect harness-visible artifacts on clips dominated by either tail of the turn-duration distribution.

**Wedge / snowplow detection and fallback**

A simple heuristic, no new ML required, derived from signals already computed:

```
wedge_likely = (
    hip_tilt_amplitude  > T_hip_min      AND   # turn shape IS present
    com_lateral_amplitude < T_com_max   AND   # but CoM doesn't excurse
    stance_measurable                         # need ankles to make this call
)
```

Logic: a carving turn produces both hip rotation *and* lateral CoM displacement (the body moves into the turn). A wedge / snowplow turn produces hip rotation *without* much lateral CoM excursion — the skier rotates over wedged skis without committing weight outward. Thresholds `T_hip_min`, `T_com_max` are calibrated in the harness (entries 17–18, below).

When `wedge_likely = true`:
- Primary CoM-based segmentation downgrades to lower confidence.
- `hip_tilt` becomes the primary segmentation signal for that segment.
- `QualityReport.wedge_likely = true` is surfaced to the user as "Some metrics are less reliable — video appears to show wedge/snowplow skiing."
- Edge-engagement metrics that assume carved turns are marked unreliable before they reach scoring or the LLM judge.

Literature reports 0.452 recall on snowplow even with IMU; monocular won't beat that. The honest move is to detect the condition and inform the user, not to pretend the carved-turn detector still works.

**Sign validation**
- Validate left/right polarity-to-direction mapping against the hand-labeled golden clip from Phase 0 before flipping any production code.

---

## Phase 6 — Metrics-only LLM judge

Last. Replace deterministic rule-tip output with an LLM judge that receives structured analysis data only. The main product bet is that if the metrics are accurate enough, the LLM can judge them better than a growing pile of handcrafted coaching rules.

**LLM input**
- Run-level score, `score_reliability`, `score_counts_for_progress`, and `quality_warnings`.
- Per-turn metrics, phase metrics, cool-moment timestamps, and turn-boundary reliability rollups.
- Stance, wedge, low-confidence, short-segment, and video-quality flags.
- No raw video, frame stream, or prompt-time visual inspection for v1. The overlay and photos remain user-visible evidence, but the judge should be grounded in the structured metrics.

**LLM output**
- Keep the existing AI-coaching artifact shape where possible (`coach_summary`, `coaching_points`, `additional_observations`, optional `recommended_drill_id`) to avoid unnecessary UI churn.
- Label the output as LLM judge feedback, not rule-generated truth.
- Include quality caveats when `score_reliability` is `limited` or `insufficient`.
- If the LLM is unavailable, show score, metrics, overlay, and a clear "LLM judge unavailable" state. Do not fall back to deterministic coaching tips.

Cross-language sync and drill-library duplication are handled by the global "Cross-language contract changes" section at the top of this document.

---

## Calibration — what the harness pins (not the plan)

Numbers in this document are starting points. The harness sets final values:

| # | Parameter | Starting value | Pinned by |
|---|-----------|----------------|-----------|
| 1 | Landmark smoothing cutoff | 6 Hz | Residual analysis on harness clips |
| 2 | Low-confidence threshold τ | 0.3 | Score-reliability and metric-reliability calibration on golden clips |
| 3 | `find_peaks` prominence floor | 0.01 (current `_MIN_AMPLITUDE_COM`) | Sweep against ground-truth turn counts |
| 4 | Short-segment threshold | 3 s (~60–90 samples) | Edge-effect error magnitude vs. segment length |
| 5 | Gap-fill max length | 10 frames | Visual inspection of interpolated poses |
| 6 | Long-turn flag threshold | 5 s | Observed turn-duration distribution on harness clips |
| 7 | Phase boundary percentages | 25 / 50 / 25 | Any clip where edge-change events are visually identifiable |
| 8 | MediaPipe + ultralytics versions + execution backend | (pin in root `pyproject.toml` + `uv.lock`) | Test environment definition; transitive determinism via lockfile |
| 9 | Stage-2 fine-tuning window | ±0.3 s | Harness measures stage-2 boundary accuracy vs. window width |
| 10 | Stance visibility threshold | 0.30 (current `_BOTH_VISIBLE_FRACTION`) | Stance-metric reliability and warning calibration |
| 11 | PTS full-stream validation + dt sanity band | 0.25× to 4× native fps | Broken-metadata guard; loose by design — VFR is supported, not rejected |
| 12 | `find_peaks` distance floor | 0.5 s × `analysis_fs` | Short-turn recall on harness |
| 13 | State-machine reset window | 1.5 s of low-confidence | False alternation rejection rate |
| 14 | Phase 5 event mapping | Undecided (A/B/C) | Compare against golden hand-labeled switches; pick winner before production rewrite |
| 15 | Nyquist safety margin | 1.25× (use nominal cutoff above this) | Filter quality vs. lost cutoff range trade |
| 16 | Low-Nyquist fallback cutoff | 0.2 × `analysis_fs` (= 40% of Nyquist) | Pre-aliasing trade on subsampled clips |
| 17 | `T_hip_min` (wedge detector) | 0.5 × current hip-tilt amplitude floor | False-positive rate of `wedge_likely` flag |
| 18 | `T_com_max` (wedge detector) | 0.5 × `_MIN_AMPLITUDE_COM` | False-negative rate of `wedge_likely` flag on hand-labeled wedge clips |
| 19 | Switch-acceptance temporal gate | 0.3 s < Δt < 5.0 s (Frontiers starting values) | Boundary FP/FN rates on golden clips |

---

## What's deliberately not in the plan

- **Multi-signal segmentation fusion**. Literature prefers single-signal + extrema + zero-phase filtering. Adding signals to a zero-crossing detector is the wrong architectural shape.
- **Discipline-specific calibration** (racer vs. beginner). No labeled corpus available; phase-aware scoring (Phase 4) captures most of the value with what's derivable.
- **Full per-metric uncertainty propagation**. Phase 2's score-reliability and metric-reliability flags capture the practical value needed for the UI and LLM judge. Full propagation is weeks of work for diminishing return.
- **4-phase scoring**. Needs edge-change detection that's not reliable from monocular video. 3-phase is the honest derivable version.
- **Multi-view stereo fusion** to fix MediaPipe z-axis noise. Not feasible for user-uploaded phone footage. Phase 2's "drop z" is the monocular-honest equivalent.
- **Multi-class viewpoint classifier** (side-on / follow-cam / oblique). Separate research task; the current ankle-visibility heuristic is only strong enough to be wrapped into a binary `stance_measurable` signal, no more.
- **Replacing MediaPipe entirely**. The Apple Vision backend (`--pose-engine vision`) already exists; architecture supports adding more.

---

## What's deliberately unchanged

- **Job lifecycle**: Supabase `jobs` row, R2 multipart upload, worker polling, artifact split (Supabase Storage for JSON/CSV/photos, R2 for `overlay.mp4`).
- **Auth model**: middleware rules, guest-vs-signed-in routes, service-role-key writes server-side only.
- **Subprocess isolation**: worker still shells out to `run.py`; pipeline crashes can't kill the worker loop.
- **LLM judge execution**: still optional and non-fatal for worker completion. If unavailable, the product shows metrics, score, overlay, and an explicit unavailable state rather than falling back to rule tips.
- **Private product mode**: public web deploy, launch, and announcement work are paused. Local/private web usage and the MacBook worker processing loop remain in scope.
- **i18n flow**: preferred language on `jobs.config`, passed to the LLM judge.
- **Design system**: Carved Arc visual language untouched.

---

## Honest limits — what the system still cannot do after this work

- Match IMU-grade turn detection (0.995 P/R on carved turns). Monocular phone footage at 15–30 fps with ~56 mm joint noise has a hard ceiling below that.
- Reliably reconstruct true 3D joint angles from a single camera. The plan acknowledges this by using 2D + signed lateral coordinates as primary and demoting 3D-derived quantities to diagnostics.
- Distinguish a racer's 45° edge angle from a beginner's 45° (no skill-level signal in the system). Phase-aware scoring is the honest derivable approximation.
- Detect true edge-change events monocularly. The 3-phase model uses heuristic % boundaries.
- Classify camera viewpoint as a multi-class field. Only the binary "is stance measurable" signal is supported.
- Achieve good performance on follow-cam footage. The plan gates affected metrics rather than fixing them — that's an architectural limit, not a tuning issue.

---

## Sequencing summary

```
Phase −1  Dependency manifest               (blocking; before everything else)
Phase 0   Regression harness                (blocking; before any algorithm change)
Phase 1   Local correctness + PTS           (parallel-safe with Phase 2)
Phase 2   Cheap wins                        (parallel-safe with Phase 1)
Phase 3   Zero-phase smoothing              (depends on Phases 0, 1, manifest)
Phase 4   Turn-phase-aware scoring          (depends on Phase 3)
Phase 5   Segmenter rewrite                 (depends on Phases 0, 1, 3)
Phase 6   Metrics-only LLM judge            (depends on Phase 4 — last)
```

The biggest single win is almost certainly Phase 3 (zero-phase filtering): every downstream metric improves and several "noisy metric" complaints likely dissolve. Next is Phase 2's skeletal-refiner fix plus z-axis drop — both remove systematic biases, not random noise. Phase 5 is the structural rewrite the system needs eventually, but it's gated on the rest being in place.

---

## Expected output — what the user gets after uploading a video

After upload completes and the worker finishes processing (typical end-to-end: 1–4 minutes for a 10–30 s clip on the MacBook worker), the recap page (`/jobs/[id]`) shows:

**1. Overlay video** — the original clip with skeleton overlay, turn boundaries marked, and per-turn quality colors. Hosted on R2, signed URL valid ~15 minutes per request.

**2. Per-turn breakdown** — for each detected turn:
  - Turn index, duration, direction (left/right), peak edge angle, peak lean angle
  - 3-phase decomposition (initiation / steering / completion) with phase-specific metrics
  - "Cool moment" photo extracted from the steering apex
  - Quality score (0–100) with severity color

**3. LLM judge feedback**, grounded only in structured metrics:
  - Plain-language summary of what the measured metrics suggest
  - Specific observations tied to turns, phases, or quality flags when available
  - Optional drill recommendation with link
  - Reliability caveats when the measured evidence is limited or insufficient
  - Explicit "LLM judge unavailable" state if the LLM call fails; no rule-tip fallback

**4. Quality flags** surfaced to the user when relevant:
  - "This video's quality is too low for reliable skeleton detection, so the score may be misleading. Try a better angle, higher resolution, closer camera, steadier shot, and cleaner single-skier framing." (when `score_reliability = insufficient`)
  - "Stance metrics are unreliable — feet not clearly visible" (when `stance_measurable = false`)
  - "Some metrics are less reliable — video appears to show wedge/snowplow skiing"
  - "Some turn boundaries flagged for manual review — turn longer than 5 s detected"
  - Insufficient-quality scores remain visible on the run page but are excluded from progress trends by default.

**5. Diagnostic fields not surfaced to the user**, available in the JSON for debugging:
  - `timestamp_source` (`pts` / `frame_idx`), `analysis_fs`, `nyquist_violation`
  - Per-segment low-confidence fractions
  - Selected Phase 5 event mapping (A/B/C)
  - Filter cutoffs actually applied (in case Nyquist guard adjusted them)

### Accuracy expectations — honest framing

**No accuracy numbers are promised in this plan because the harness has not run yet.** That is by design: Phase 0 exists precisely so accuracy is measured, not claimed. The plan deliberately ships with no marketing-grade numbers attached.

The table below lists **pre-harness hypotheses to measure** — directional rankings of expected difficulty, not numeric commitments. Specific percentages will be filled in by Phase 0 against the fixture set and kept with the model notes before any public claim is made.

| Condition | Pre-harness hypothesis (to be measured, not claimed) |
|---|---|
| Carved turns, side-on or down-fall-line, good light, single skier | Best-case regime; expect turn count and boundary timing to be most accurate here. |
| Carved turns, oblique angle | Degraded vs. side-on; some lateral CoM signal projection loss. |
| Carved turns, follow-cam | Significantly degraded; stance metrics marked unreliable; turn counts may still be usable. |
| Snowplow / wedge / beginner | `wedge_likely` flagged; user warned; turn counts unreliable by hypothesis. |
| Variable terrain / mixed turn shapes | Bounded by the worst-case shape in the clip. |

**Physical / signal-source constraints that bound the best case** (cited, not derived):
- The IMU literature ceiling (~0.995 P/R on carved turns with boot-mounted sensors at 100+ Hz) is the upper bound monocular phone footage cannot beat.
- MediaPipe monocular MPJPE ~56 mm. Translating this into a specific per-joint angular error is methodology-dependent — published evaluations range from < 1° (validated against Kinovea on planar exercises) to ~26° (knee angle in unconstrained monocular). The honest statement: pose noise is large enough to make precise joint-angle judging unreliable without explicit score and metric reliability flags.
- The Frontiers paper's own snowplow result (0.452 recall) for a vastly better signal source than ours.

**Per-joint angle uncertainty**: the LLM judge should not treat precise corrections ("increase edge angle from 8° to 15°") as reliable when the underlying metric is weak. When `score_reliability` is `limited` or `insufficient`, the product keeps the score visible but tells the user the result may be misleading and excludes insufficient scores from progress trends. This is the practical answer to the angular-error problem until either a measured per-joint error table exists (Phase 0 deliverable) or a stereo / multi-view setup becomes available.

**What we explicitly do *not* claim**:
- A flat accuracy number across all camera angles. The plan rejects that framing — angle and conditions dominate the result.
- Better-than-IMU accuracy on carved turns. Physically impossible from monocular phone footage.
- Reliable judging on follow-cam footage of low-skill skiing. We keep the score visible, gate weak metrics, and inform the user rather than pretending the evidence is strong.
- Discipline-aware scoring (racer vs. beginner). The phase-aware scoring is the honest derivable proxy.

**What the harness will let us report internally first (after it runs)**:
- Measured turn-count accuracy and boundary RMSE per the categories in the table above.
- Measured false-positive rate of each movement-quality label and metric-derived warning, pre- and post-Phase 2.
- Measured `low_confidence_fraction` distributions across the fixture set.
- A per-condition confidence interval, not a single number.

The product-honest version of "what accuracy?" is: *"We show a score with its reliability, warn clearly when skeleton detection is too weak to trust, exclude insufficient-quality scores from progress trends, and report measured per-condition accuracy on a fixed test set once the harness has run."*

---

## Sources

Alpine skiing turn detection:
- [Development and Validation of a Gyroscope-Based Turn Detection Algorithm for Alpine Skiing (Frontiers / PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7739568/) — two-stage 0.5 Hz / 3.0 Hz filtering, sign-alternation state machine, turn-duration distributions (short carved: 1.04 ± 0.41 s; long carved: 2.97 ± 0.43 s).
- [Motion Analysis in Alpine Skiing: Sensor Placement and Orientation-Invariant Sensing (Sensors 2025)](https://www.mdpi.com/1424-8220/25/8/2582)
- [Influence of slope steepness, foot position and turn phase on plantar pressure in GS (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5417654/) — turn-phase decomposition.

MediaPipe / monocular pose accuracy:
- [Accuracy Evaluation of 3D Pose Reconstruction with MediaPipe (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644880/) — monocular MPJPE ~56.3 mm vs. stereo ~30.1 mm. Monocular 3D pose is noisy overall; depth-sensitive constructions (e.g., `√(x² + z²)`) should not drive primary scores.
- [Human Pose Estimation Using MediaPipe Pose and a Humanoid Model (Applied Sciences)](https://www.mdpi.com/2076-3417/13/4/2700) — BlazePose fails 12° clinical reliability threshold; inconsistent bone-length estimates.

Biomechanics signal filtering:
- [Filtering Biomechanical Signals in Movement Analysis (Sensors / PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8271607/) — Butterworth zero-phase recommended; edge-effect errors 14–170% on short non-periodic gestures.

Motion capture gap filling:
- [A Fully-Automatic Gap Filling Approach for Motion Capture Trajectories (Applied Sciences)](https://www.mdpi.com/2076-3417/11/21/9847)
- [Predicting Missing Marker Trajectories in Human Motion Data (PMC)](https://ncbi.nlm.nih.gov/pmc/articles/PMC4816448) — linear / spline are field standard for short gaps; carry-forward is not recommended.

SciPy implementation:
- [`scipy.signal.butter` — `output='sos'` recommended for numerical stability](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
- [`scipy.signal.sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)
- [`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)

OpenCV video timestamps:
- [OpenCV `CAP_PROP_POS_MSEC` / `CAP_PROP_PTS` documentation](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)
