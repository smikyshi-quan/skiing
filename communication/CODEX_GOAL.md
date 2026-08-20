# Codex `/goal`

```text
/goal Implement the web, trend, worker, and metrics-only LLM judge side of Improvements V2 on branch improvements-v2.

Before coding:
- Run: git switch improvements-v2
- Read:
  - AGENTS.md
  - CLAUDE.md
  - improvement_proposal/metrics_and_coaching_plan.md
  - communication/README.md
  - communication/backend_contract.md if Claude Code has already created it
- Create or update communication/codex_status.md with your current plan and later your final status.
- If you need Claude Code to change a Python summary field or backend artifact, write the request in communication/handoff_requests.md instead of editing Claude-owned files.

You own these files/areas:
- MVP/web/**
- MVP/worker.py
- MVP/lmstudio_coaching.py
- MVP/claude_coaching.py and MVP/gemini_coaching.py only if needed to keep shared LLM behavior consistent
- TypeScript summary parsing/rendering and score trend consumers

Do not edit these areas unless Claude Code explicitly requests it in communication/handoff_requests.md:
- technique-analysis/src/technique_analysis/**
- technique-analysis/tests/**
- Python contract models
- backend algorithm internals

Assume this backend JSON contract unless communication/backend_contract.md says otherwise:
{
  "quality_report": {
    "score_reliability": "reliable" | "limited" | "insufficient",
    "score_counts_for_progress": boolean,
    "quality_warnings": string[],
    "stance_measurable": boolean,
    "stance_visibility_fraction": number,
    "wedge_likely": boolean
  }
}

Primary deliverables:
1. Mirror the new fields in the TypeScript summary types and parser:
   - score_reliability
   - score_counts_for_progress
   - quality_warnings
   - stance_measurable
   - stance_visibility_fraction
   - wedge_likely
2. Keep parsing backward compatible:
   - missing score_reliability => "limited"
   - missing score_counts_for_progress => score_reliability !== "insufficient"
   - missing quality_warnings => []
   - unknown quality_warnings codes must not crash rendering
3. Recap UI behavior:
   - Show the numeric score whenever score exists.
   - Remove or bypass any current behavior that hides score for insufficient reliability.
   - When score_reliability is "insufficient", show this warning clearly:
     "This video's quality is too low for reliable skeleton detection, so the score may be misleading. Try a better angle, higher resolution, closer camera, steadier shot, and cleaner single-skier framing."
   - For "limited", show a softer reliability caveat without hiding the score.
4. Trends/profile/history behavior:
   - Exclude runs with score_counts_for_progress = false from averages, progress trends, and history trend calculations by default.
   - Keep those runs visible in run lists/history; mark them as excluded or unreliable instead of deleting/hiding them.
5. LLM judge behavior:
   - Update lmstudio_coaching.py so the prompt/input uses structured metrics only.
   - Include score, score_reliability, quality_warnings, per-turn metrics, phase metrics, stance/wedge flags, and available quality flags.
   - Do not dump the entire raw summary if it includes irrelevant internals.
   - Do not use deterministic rule tips as fallback user-facing coaching.
   - If the LLM call fails, keep the job successful and expose an "LLM judge unavailable" state.
6. UI language:
   - Replace "Score unavailable" for low-quality videos with score-visible reliability warning behavior.
   - Replace deterministic "coaching tips" assumptions with "LLM judge feedback" or equivalent product language where user-facing.
   - Keep i18n behavior intact; add/update translation keys where needed.

Quality warning codes to recognize:
- low_pose_confidence
- insufficient_skeleton_detection
- stance_not_measurable
- wedge_likely
- short_clip
- low_boundary_reliability
- tracking_loss
- follow_cam_degraded

Web acceptance criteria:
- A run with score_reliability = "insufficient" still displays its score.
- The insufficient warning copy is visible on the run recap.
- The same run is excluded from progress/trend calculations.
- A run with score_reliability = "limited" displays score plus a softer caveat and still counts for progress unless score_counts_for_progress is false.
- Existing summaries without the new fields still render.
- LLM failure does not fail the worker job and does not show rule-tip fallback as if it were LLM output.

Verification to run:
- From MVP/web: npm run build
- From MVP/web: npx tsc --noEmit if useful
- If feasible, test with a mocked/fixture summary containing each score_reliability value.

Final handoff:
- Update communication/codex_status.md with changed files and verification results.
- If the final web behavior depends on backend field names, record those assumptions in communication/codex_status.md and check communication/backend_contract.md before finalizing.
```

