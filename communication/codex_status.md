# Codex status — Improvements V2 web / worker / LLM judge

Branch: `improvements-v2`. Owner: Codex (web product surface, trend behavior, worker + LM Studio judge path, TypeScript consumers).

## Implemented

- TypeScript summary contract now reads `quality_report` first and falls back to legacy `quality`.
- Mirrored Improvements V2 quality fields:
  `score_reliability`, `score_counts_for_progress`, `quality_warnings`,
  `stance_measurable`, `stance_visibility_fraction`, `wedge_likely`.
- Backward-compatible defaults:
  missing `score_reliability` => `limited`;
  missing `score_counts_for_progress` => `score_reliability !== "insufficient"`;
  missing `quality_warnings` => `[]`;
  unknown warning codes are tolerated.
- Recap UI always shows a numeric score when one exists, including
  `score_reliability = "insufficient"`.
- Insufficient-quality recap warning uses the required copy:
  "This video's quality is too low for reliable skeleton detection, so the score may be misleading. Try a better angle, higher resolution, closer camera, steadier shot, and cleaner single-skier framing."
- Limited-quality recap state shows a softer caveat while keeping the score visible.
- Dashboard, profile, archive, score trend card, season averages, best-run summaries,
  and previous-score deltas exclude runs whose summary has
  `score_counts_for_progress = false`.
- Run lists keep excluded scores visible and label them as excluded from progress trends.
- Run detail page labels feedback as LLM judge feedback and shows a clear
  "LLM judge unavailable" state for done jobs when judge output is missing or marked unavailable.
- Dashboard latest insight / practice focus now uses `ai_coaching` LLM judge artifacts
  instead of deterministic `summary.coaching_tips`.
- LM Studio prompt path now sends a compact `improvements_v2_metrics_only` payload
  instead of dumping the full raw summary. The payload includes score, reliability,
  progress inclusion, quality warnings, stance/wedge flags, video quality fields,
  tracking segments, per-turn metrics, optional phase metrics, turn-boundary reliability,
  and cool-moment timestamps.
- Worker keeps jobs successful when the local LLM fails and uploads an explicit
  `ai_coaching.json` artifact with `judge_status = "unavailable"`; no rule-tip fallback
  is shown as LLM output.

## Changed files

- `MVP/web/lib/analysis-summary.ts`
- `MVP/web/lib/server-job-data.ts`
- `MVP/web/lib/i18n.ts`
- `MVP/web/components/score-trend-card.tsx`
- `MVP/web/components/archive-runs-client.tsx`
- `MVP/web/app/page.tsx`
- `MVP/web/app/profile/page.tsx`
- `MVP/web/app/jobs/page.tsx`
- `MVP/web/app/jobs/[id]/page.tsx`
- `MVP/web/app/api/jobs/[id]/route.ts`
- `MVP/lmstudio_coaching.py`
- `MVP/worker.py`

## Verification

- `npx tsc --noEmit` from `MVP/web` — passed.
- `npm run build` from `MVP/web` — passed after allowing network access for Next/Google Fonts. First sandboxed attempt failed only because `fonts.googleapis.com` was blocked.
- `python3 -m py_compile MVP/lmstudio_coaching.py` — passed.
- `python3 -m py_compile MVP/worker.py` — passed.
- Metrics-only payload smoke test against `MVP/web/lib/sample-data/summary.json` — passed:
  `judge_input_version = "improvements_v2_metrics_only"`,
  missing reliability defaults to `limited`,
  `score_counts_for_progress = true`,
  4 turns included,
  deterministic `coaching_tips` not included.
- `npm run dev` from `MVP/web` — running at `http://localhost:3000` after
  allowing the dev server to bind the port.
- `curl -I http://localhost:3000/sample-analysis` — returned `HTTP/1.1 200 OK`.

## Contract notes

- `communication/backend_contract.md` was not present when this status was written.
- Web behavior follows `communication/README.md` and the current backend worktree shape:
  summaries may expose the V2 fields under `quality_report`, with legacy `quality`
  still tolerated.
