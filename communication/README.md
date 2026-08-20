# Improvements V2 Agent Coordination

Both agents must work on branch `improvements-v2`.

```bash
git switch improvements-v2
```

The product source of truth is:

- `improvement_proposal/metrics_and_coaching_plan.md`
- `AGENTS.md`
- `CLAUDE.md`

## Communication Rules

- Use this folder only for coordination notes, handoffs, blockers, and final status.
- Do not edit another agent's status file. Append to your own file instead.
- If you need the other agent to change something, write it in `handoff_requests.md` with:
  - owner requested: `claude-code` or `codex`
  - file/interface affected
  - exact requested change
  - whether it is blocking
- Before finalizing, each agent must read:
  - `communication/handoff_requests.md` if it exists
  - the other agent's status file if it exists

## Ownership Boundary

Claude Code owns the Python analysis pipeline and summary contract.
Codex owns the web product surface, trend behavior, worker/LLM judge path, and TypeScript consumers.

Do not edit files owned by the other agent unless the other agent has explicitly requested it in `handoff_requests.md`.

## Shared Contract

The summary JSON should expose these snake_case fields under the run quality object used by the web dashboard, preferably `quality_report`:

```json
{
  "score_reliability": "reliable",
  "score_counts_for_progress": true,
  "quality_warnings": [],
  "stance_measurable": true,
  "stance_visibility_fraction": 0.85,
  "wedge_likely": false
}
```

Allowed `score_reliability` values:

- `reliable`
- `limited`
- `insufficient`

Allowed `quality_warnings` codes:

- `low_pose_confidence`
- `insufficient_skeleton_detection`
- `stance_not_measurable`
- `wedge_likely`
- `short_clip`
- `low_boundary_reliability`
- `tracking_loss`
- `follow_cam_degraded`

Unknown warning codes must be tolerated by the web UI.

## Product Rules

- Always show a numeric score when analysis completes and a score exists.
- Never hide the score only because reliability is insufficient.
- When `score_reliability = "insufficient"`, show a strong warning that the result may be misleading.
- Exclude insufficient scores from progress/history/trend calculations by default.
- The LLM is a metrics-only judge. Do not rely on deterministic coaching tips as fallback user-facing advice.
- If the LLM call fails, the job should still complete and the UI should show `LLM judge unavailable`.

