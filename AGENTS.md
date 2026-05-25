# Repository Guidelines

## Project Structure & Module Organization

This repository is a monorepo for SkiCoach AI. `technique-analysis/` contains the Python CV pipeline under `src/technique_analysis/`, including pose extraction, smoothing, metrics, turn segmentation, coaching rules, and overlay rendering. `MVP/` contains worker and orchestration scripts such as `run.py`, `worker.py`, and coaching backends. `MVP/web/` is the Next.js 14 App Router frontend. `supabase/migrations/` holds Postgres schema and RLS changes. Generated analysis outputs live under `technique-analysis/artifacts/` and should not be committed.

## Build, Test, and Development Commands

Run web commands from `MVP/web/`:

```bash
npm run dev              # start Next.js locally on port 3000
npm run build            # production build with type checking
npm start                # serve a built app
npx tsc --noEmit         # standalone TypeScript check
npm run backfill-scores  # recompute job scores from stored summaries
```

Run pipeline and worker commands from the repo root:

```bash
python MVP/run.py <video.mp4>              # analyze one video
python MVP/run.py <video.mp4> --no-overlay # skip overlay rendering
python MVP/worker.py --once                # process one queued job
python MVP/worker.py --recover             # requeue stale running jobs
```

## Coding Style & Naming Conventions

Use TypeScript and React conventions in `MVP/web`: PascalCase components, camelCase functions and variables, and the `@/*` path alias for imports from the web root. Keep server-only Supabase writes in server routes using the service-role client. Python code uses snake_case modules, functions, and variables, with dataclasses and shared contracts centralized in `technique_analysis/common/contracts/models.py`.

## Testing Guidelines

No formal test runner is currently configured. Before submitting changes, run `npm run build` or `npx tsc --noEmit` for frontend changes. For pipeline changes, run `python MVP/run.py <sample-video>` when a local video is available and inspect the generated summary and overlay artifacts. Keep Python contract changes synchronized with `MVP/web/lib/analysis-summary.ts`.

## Commit & Pull Request Guidelines

Recent commits use short imperative messages such as `Improve app UX, language support, and guest flow` and `Prevent truncated upload analysis failures`. Follow that style: start with a verb, summarize the behavior change, and avoid noisy prefixes. Pull requests should describe the user-facing change, list verification commands, link related issues, and include screenshots or video clips for UI or overlay changes.

## Security & Configuration Tips

Never commit local secrets. Use `MVP/.env.worker.example` and `MVP/web/.env.local.example` as templates. Preserve the R2/Supabase storage fallback behavior for older jobs, and keep LM Studio coaching failures non-fatal so worker jobs can still complete without AI coaching.
