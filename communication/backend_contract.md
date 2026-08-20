# Backend contract — Improvements V2 (Claude Code → Codex)

Branch: `improvements-v2`. Owner: Claude Code (Python analysis + summary contract).

This document is the source of truth for the new summary JSON fields that the
web side (Codex) is expected to consume. Read this before wiring the TypeScript
parser or any UI rendering.

## Where the new fields live in `summary.json`

The new fields live on the existing `QualityReport` object. The same object is
emitted under **two** top-level keys so both the legacy reader (`quality`) and
the Improvements V2 contract reader (`quality_report`) parse cleanly:

```json
{
  "run_id": "...",
  "video_metadata": { ... },
  "turns": [ ... ],
  "segments": [ ... ],
  "coaching_tips": [ ... ],

  "quality_report": {
    "overall_pose_confidence_mean": 0.82,
    "overall_pose_confidence_min":  0.41,
    "low_confidence_fraction":      0.08,
    "viewpoint_warning":            null,
    "jitter_score_mean":            0.02,
    "warnings":                     [],
    "resolved_max_fps":             20.0,
    "resolved_max_dimension":       720,

    "score_reliability":            "reliable",
    "score_counts_for_progress":    true,
    "quality_warnings":             [],
    "stance_measurable":            true,
    "stance_visibility_fraction":   0.85,
    "wedge_likely":                 false
  },

  "quality":     { ... same payload as quality_report ... },

  "diagnostics": {
    "timestamp_source":                       null,
    "timestamp_warning":                      null,
    "analysis_fs":                            null,
    "end_timestamp_s":                        null,
    "nyquist_violation":                      [],
    "filter_cutoffs_applied":                 {},
    "per_segment_low_confidence_fraction":    [],
    "event_mapping":                          null,
    "boundary_reliability_by_turn":           {},
    "stage_2_refinement_counts":              {},
    "rejected_candidates":                    []
  }
}
```

The `mvp_summary.json` written by `MVP/run.py` mirrors the same `quality` /
`quality_report` blocks (compact form) so the worker and LLM judge see the new
fields without re-parsing the full `summary.json`.

## Field definitions

All field names are `snake_case` per the goal.

| Field | Type | Allowed values | Default |
|---|---|---|---|
| `score_reliability` | string | `"reliable"` \| `"limited"` \| `"insufficient"` | `"limited"` |
| `score_counts_for_progress` | boolean | — | `true` |
| `quality_warnings` | string[] | see warning codes below | `[]` |
| `stance_measurable` | boolean | — | `true` |
| `stance_visibility_fraction` | number ∈ [0, 1] | — | `1.0` |
| `wedge_likely` | boolean | — | `false` |

### Invariants the backend guarantees

1. `score_counts_for_progress === (score_reliability !== "insufficient")`.
2. A numeric score (per-frame `overall_score`, per-turn `quality_score`) is
   **always preserved** when scoring has enough data to produce one. Reliability
   is reported separately; the score is never nulled because reliability is
   insufficient.
3. `quality_warnings` is always a list (possibly empty), never `null`.
4. The same `QualityReport` payload is emitted at both `summary.quality` and
   `summary.quality_report`. They are *not* a shallow alias — they are the same
   serialized payload. The web side may read either, but the
   Improvements-V2-canonical key is `quality_report`.

### Backward compatibility for older / partial runs

- Summaries written before this branch shipped do **not** have the new fields.
  The fields all carry safe defaults on the dataclass, so a re-run will populate
  them. Pre-existing JSON files on disk will simply lack the keys; the Codex
  TS parser should already tolerate that per the CODEX_GOAL fallback rules.
- If a code path produces a `QualityReport` without going through
  `compute_reliability()` (none currently do, but defensively): the defaults
  yield `score_reliability="limited"`, `score_counts_for_progress=true`,
  `quality_warnings=[]`, `stance_measurable=true`,
  `stance_visibility_fraction=1.0`, `wedge_likely=false` — matching the
  CODEX_GOAL fallback contract.

## Warning codes actually emitted by this branch

| Code | Trigger |
|---|---|
| `low_pose_confidence` | `overall_pose_confidence_mean < 0.50` OR `low_confidence_fraction > 0.30` |
| `insufficient_skeleton_detection` | `overall_pose_confidence_mean < 0.35` OR `low_confidence_fraction > 0.60` |
| `stance_not_measurable` | `stance_visibility_fraction < 0.30` (both ankles visible together in fewer than 30% of frames) |
| `wedge_likely` | hip_tilt std > 0.5° AND com_shift_x std < 0.005 m AND `stance_measurable` |
| `short_clip` | `video_duration_s < 4.0` |
| `low_boundary_reliability` | 0 turns on a ≥ 4s clip with usable pose confidence, OR turns detected but their average per-turn pose confidence < 0.50 |
| `tracking_loss` | More than one `TrackingSegment` (multi-skier or tracker re-lock) |

Unknown codes must be tolerated by the web UI per `communication/README.md`.

## Reliability classifier

```
if insufficient_skeleton_detection in warnings:
    score_reliability = "insufficient"
elif 0 turns AND clip ≥ 4s AND low_pose_confidence in warnings:
    score_reliability = "insufficient"
elif any other warning in warnings:
    score_reliability = "limited"
else:
    score_reliability = "reliable"

score_counts_for_progress = (score_reliability != "insufficient")
```

## Warning codes intentionally **not** emitted by this branch

- `follow_cam_degraded` — would require a camera-motion / follow-cam detector.
  No such signal exists in the current pipeline. The string is still part of
  the contract (the web UI must tolerate it), but this branch never emits it.
  Recommend implementing alongside the Phase-5 segmenter rewrite, where camera
  / tracking-segment statistics are already gathered.

## Internal diagnostics (Tier 2 — not for UI)

`summary.diagnostics` is a `DiagnosticsBundle` whose fields are populated by
later Improvements V2 phases (Phase 1 PTS, Phase 3 Nyquist guard, Phase 5
event-mapping and per-turn boundary reliability). This pass declares the schema
with empty/`None` defaults so the JSON shape is stable across upcoming phases.

The web side **must not** read from `diagnostics` for any rendered behavior.
Treat it as `unknown` / `any`.

## What this branch did **not** change

- `CoachingTip.evidence_reliability` — explicitly **not** added (per goal item 8).
- Deterministic coaching tip generation — preserved but not promoted. The
  metrics-only LLM judge is Codex's domain. If the LLM judge fails, the worker
  is expected to keep the job successful and surface "LLM judge unavailable"
  without falling back to rule-tip output (CODEX_GOAL.md item 5 / 6).

## Known missing signals / TODOs

- `follow_cam_degraded` warning code — pending camera-motion signal.
- `diagnostics.timestamp_source` / `analysis_fs` / `end_timestamp_s` — pending
  Phase 1 PTS-aware iterator.
- `diagnostics.nyquist_violation` / `filter_cutoffs_applied` — pending Phase 3.
- `diagnostics.event_mapping` / `boundary_reliability_by_turn` /
  `stage_2_refinement_counts` / `rejected_candidates` — pending Phase 5.
- Wedge / boundary-reliability thresholds are pre-harness starting values
  (Phase-0 calibration entries 17, 18 and the turn-confidence floor used here).
  Numbers will move once the harness runs.

## Verification (this branch)

Run from the repo root:

```bash
PYTHONPATH=technique-analysis/src python3 -c "
from technique_analysis.common.contracts.models import QualityReport, DiagnosticsBundle, TechniqueRunSummary
from technique_analysis.common.quality import compute_reliability, compute_stance_visibility_fraction, compute_wedge_likely
print('contract OK')
"
```

End-to-end `python MVP/run.py <sample-video>` not verified on this machine — no
local sample video available. The narrowest available checks (model imports,
dataclass defaults, `compute_reliability` truth table for every documented
warning code, summary `as_dict()` JSON roundtrip with both `quality` and
`quality_report` keys present, `MVP/run.py` import) all pass.
