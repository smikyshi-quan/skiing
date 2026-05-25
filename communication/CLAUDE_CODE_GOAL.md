# Claude Code `/goal`

```text
/goal Implement the backend analysis and summary-contract side of Improvements V2 on branch improvements-v2.

Before coding:
- Run: git switch improvements-v2
- Read:
  - AGENTS.md
  - CLAUDE.md
  - improvement_proposal/metrics_and_coaching_plan.md
  - communication/README.md
- Create or update communication/claude_status.md with your current plan and later your final status.
- If you need Codex to change a web/worker/LLM consumer, write the request in communication/handoff_requests.md instead of editing Codex-owned files.

You own these files/areas:
- technique-analysis/src/technique_analysis/**
- technique-analysis/tests/**
- Python contract models, especially technique_analysis/common/contracts/models.py
- MVP/run.py only when needed to preserve or emit summary JSON/artifacts
- root Python dependency manifest/lockfile only if required by the backend work

Do not edit these areas unless Codex explicitly requests it in communication/handoff_requests.md:
- MVP/web/**
- MVP/worker.py
- MVP/lmstudio_coaching.py
- MVP/claude_coaching.py
- MVP/gemini_coaching.py

Primary deliverables:
1. Add a backend summary contract for score reliability:
   - score_reliability: "reliable" | "limited" | "insufficient"
   - score_counts_for_progress: bool
   - quality_warnings: list[str]
   - stance_measurable: bool
   - stance_visibility_fraction: float
   - wedge_likely: bool
2. Put these fields on the existing public quality-report/run-quality object consumed by summary JSON, preferably quality_report.
3. Keep JSON field names snake_case.
4. Add safe defaults for older/partial analysis paths:
   - score_reliability defaults to "limited" when analysis succeeds but confidence cannot be fully assessed.
   - score_counts_for_progress is false only when score_reliability is "insufficient".
   - quality_warnings defaults to [].
   - stance_measurable defaults to true only if the existing visibility signal supports that; otherwise use false with stance_not_measurable.
   - wedge_likely defaults to false unless the heuristic is implemented and triggered.
5. Always preserve a numeric score when analysis completes and scoring has enough data to produce one. Do not hide/null the score because reliability is insufficient.
6. Implement reliability using currently available signals first:
   - pose/landmark confidence
   - low-confidence frame/window fraction
   - stance visibility
   - short segment or too-few-turn evidence
   - tracking loss / cross-skier degradation if available
   - wedge/follow-cam degradation if available
   If a signal is not available, do not invent it; leave that warning absent and document the gap in communication/claude_status.md.
7. Add or preserve an internal DiagnosticsBundle for Tier 2 diagnostic fields so debug data does not accidentally become public UI contract.
8. Do not add CoachingTip.evidence_reliability.
9. Do not implement deterministic coaching tips as the main user-facing product path.

Quality warning codes to emit when applicable:
- low_pose_confidence
- insufficient_skeleton_detection
- stance_not_measurable
- wedge_likely
- short_clip
- low_boundary_reliability
- tracking_loss
- follow_cam_degraded

Backend acceptance criteria:
- A completed run summary includes the new quality fields.
- score_counts_for_progress is exactly false when score_reliability is "insufficient"; true otherwise.
- Existing run.py output remains backward compatible for current consumers.
- No public CoachingTip evidence_reliability field is introduced.
- Any internal diagnostics live under diagnostics or another clearly internal object.

Verification to run if feasible:
- python MVP/run.py <local-sample-video> --no-overlay
- If no local sample video exists, run the narrowest import/type/smoke check available and document what could not be verified.

Final handoff:
- Write communication/backend_contract.md with:
  - final JSON field location
  - sample summary JSON fragment
  - warning codes actually emitted
  - known missing signals or TODOs
- Update communication/claude_status.md with changed files and verification results.
```

