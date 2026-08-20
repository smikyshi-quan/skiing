# Claude Code status — Improvements V2 backend

Branch: `improvements-v2`. Owner: Claude Code (Python analysis pipeline + summary contract).

## Plan

1. Extend `technique_analysis.common.contracts.models`:
   - New `DiagnosticsBundle` dataclass (Tier 2 internal-only fields).
   - New `QualityReport` fields (Tier 1 public):
     `score_reliability`, `score_counts_for_progress`, `quality_warnings`,
     `stance_measurable`, `stance_visibility_fraction`, `wedge_likely`.
   - New `TechniqueRunSummary.diagnostics: DiagnosticsBundle` field.
2. Add `technique_analysis.common.quality` module with helpers:
   - `compute_stance_visibility_fraction(poses)`
   - `compute_wedge_likely(metrics, stance_measurable)`
   - `compute_reliability(...)` → `(score_reliability, score_counts_for_progress, warning_codes)`.
3. Wire into the orchestrator `_build_quality_report` so a completed run populates
   the new fields and emits warning codes from the documented set.
4. Update `MVP/run.py`'s `mvp_summary.json` so the worker/LLM judge sees the new fields.
5. Emit the new contract under both `quality` (legacy) and `quality_report` (new
   key from `communication/README.md`) in the summary JSON, so the web side can
   migrate without breaking older readers.

## Safe defaults (when signals are partially available)

- `score_reliability` defaults to `"limited"` if a quality report is built but
  signals aren't strong enough to declare reliable or insufficient.
- `score_counts_for_progress = (score_reliability != "insufficient")`.
- `quality_warnings = []` when no codes fire.
- `stance_measurable` defaults to `True`; flipped to `False` only when the
  ankle-visibility heuristic measures stance visibility below threshold.
- `stance_visibility_fraction` defaults to `1.0` (matches "no degradation
  observed") when not computable; the warning code is only emitted when the
  fraction is actually low.
- `wedge_likely` defaults to `False`; only set to `True` when both required
  signals (hip_tilt amplitude, com_shift_x amplitude) clear conservative starting
  thresholds **and** stance is measurable.

## Warning codes wired

- `low_pose_confidence` — `overall_pose_confidence_mean < 0.5` or `low_confidence_fraction > 0.30`.
- `insufficient_skeleton_detection` — `overall_pose_confidence_mean < 0.35` or `low_confidence_fraction > 0.60`.
- `stance_not_measurable` — `stance_visibility_fraction < 0.30`.
- `wedge_likely` — heuristic from hip_tilt vs com_shift_x amplitudes (Phase 5 conservative starting thresholds).
- `short_clip` — `video_duration_s < 4.0` (1 carved turn ≈ 1–3 s).
- `low_boundary_reliability` — no turns detected on a long-enough clip with usable pose confidence, or turns detected but their average pose confidence is low.
- `tracking_loss` — `len(segments) > 1` (multi-skier or tracker re-lock).
- `follow_cam_degraded` — **not wired** in this pass. Camera-motion signal is not available from the current pipeline; see "Known gaps" below.

## Reliability classifier

- `insufficient` — `insufficient_skeleton_detection` is emitted, OR no turns were detected on a long-enough clip with low confidence.
- `limited` — any other warning code is emitted.
- `reliable` — no warning codes (other than informational viewpoint string) fire.

The numeric run/turn scores are always preserved when scoring has enough data;
reliability is reported alongside.

## Known gaps (do not invent signals)

- `follow_cam_degraded` — requires a camera-motion / follow-cam detector that
  this branch does not implement. Web should still tolerate the code when it
  appears in `quality_warnings`. Recommend implementing in the Phase-5 segmenter
  pass where tracking/camera-motion stats are already available.
- Per-turn `boundary_reliability_by_turn` (Phase 5) is declared on
  `DiagnosticsBundle` but left empty by this pass; Phase 5 will populate it.
- `nyquist_violation`, `filter_cutoffs_applied`, `event_mapping`,
  `per_segment_low_confidence_fraction`, `stage_2_refinement_counts`,
  `rejected_candidates` — declared on `DiagnosticsBundle` for forward compat
  with Phase 3/5 but populated as empty defaults in this pass.

## Changed files

- `technique-analysis/src/technique_analysis/common/contracts/models.py`
- `technique-analysis/src/technique_analysis/common/quality.py` (new)
- `technique-analysis/src/technique_analysis/free_ski/pipeline/orchestrator.py`
- `MVP/run.py`
- `communication/backend_contract.md` (new — final handoff)

## Verification

End-to-end `python MVP/run.py <sample-video>` was **not** run — no local sample
video available on this machine (`find . -name '*.mp4' -o -name '*.mov'` is
empty outside `.git/`). The narrowest available checks were run instead:

1. **Import smoke test** — `technique_analysis.common.contracts.models` and
   `technique_analysis.common.quality` import cleanly.
2. **Dataclass defaults** — `QualityReport()` constructed with only the legacy
   required args yields the CODEX_GOAL fallback defaults
   (`score_reliability="limited"`, `score_counts_for_progress=True`,
   `quality_warnings=[]`, `stance_measurable=True`,
   `stance_visibility_fraction=1.0`, `wedge_likely=False`).
3. **`DiagnosticsBundle` defaults** — all fields default to `None` / empty
   collections; `as_dict()` serializes cleanly.
4. **`compute_reliability` truth table** — manually exercised each documented
   warning code (low_pose_confidence, insufficient_skeleton_detection,
   stance_not_measurable, wedge_likely, short_clip, low_boundary_reliability,
   tracking_loss). Every documented case classifies into the right tier and
   `score_counts_for_progress` matches `(reliability != "insufficient")`.
5. **`compute_stance_visibility_fraction` / `compute_wedge_likely`** — exercised
   with synthetic frame poses / metrics. Side-view → 0.0, front-view → 1.0,
   wedge synthetic clip (large hip swing, tiny CoM) → `True`, carved synthetic
   clip → `False`. Gated correctly by `stance_measurable=False`.
6. **`summary.as_dict()` roundtrip** — produces top-level keys `quality` AND
   `quality_report` with identical payloads, plus `diagnostics`. New fields
   serialize as snake_case JSON booleans / strings as documented.
7. **`MVP/run.py`** — imports cleanly.

The full pipeline (`MVP/run.py <video>`) needs to be exercised by whoever has a
local sample available; the contract layer is verified independently.

## Final status

Done. Handoff document at `communication/backend_contract.md`. No changes are
requested from Codex via `handoff_requests.md` — the new fields and warning
codes match the cross-agent contract in `communication/README.md` and the
parser assumptions in `communication/CODEX_GOAL.md`.

