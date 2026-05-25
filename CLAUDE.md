# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a monorepo for **SkiCoach AI** — a video-based ski technique analysis MVP. Three top-level workspaces:

- `technique-analysis/` — Python package (`technique_analysis`, namespaced under `src/`) implementing the CV pipeline: pose extraction → smoothing → turn segmentation → per-frame/per-turn metrics → rule-based coaching tips → overlay rendering.
- `MVP/` — Glue layer. `run.py` runs the analysis pipeline on a single video. `worker.py` is a local long-running worker that polls Supabase for queued jobs, downloads the video from R2 (or Supabase Storage), shells out to `run.py`, generates LLM coaching, and uploads artifacts. `lmstudio_coaching.py` / `claude_coaching.py` / `gemini_coaching.py` are interchangeable coaching backends; the worker uses LM Studio.
- `MVP/web/` — Next.js 14 (App Router) front-end deployed to Vercel. Handles auth, multipart R2 uploads, job listing, and recap display.
- `supabase/migrations/` — Postgres schema (`jobs`, `artifacts` tables with RLS).

The root `vercel.json` points Vercel at `MVP/web/package.json`, so the web app is the only thing deployed; the Python worker runs on a developer's MacBook.

## Common commands

### Web app (`MVP/web/`)
```bash
npm run dev                 # next dev (port 3000)
npm run build               # next build — also runs full type-check
npm start                   # production server
npx tsc --noEmit            # standalone type-check (only TS linter in use)
npm run backfill-scores     # node script: recomputes jobs.score from summary.json artifacts
```
There is no ESLint, Prettier, or test runner configured for the web app. Type-checking via `tsc` (or the `next build` that wraps it) is the only automated check.

### Python pipeline / worker (`MVP/`)
```bash
python MVP/run.py <video.mp4>                       # one-shot analysis, writes to technique-analysis/artifacts/runs/<timestamp>_<name>/
python MVP/run.py <video.mp4> --pose-engine vision  # Apple Vision backend (macOS 14+, often faster on Apple Silicon)
python MVP/run.py <video.mp4> --no-overlay          # skip overlay video rendering
python MVP/worker.py                                # poll loop — main worker entry
python MVP/worker.py --once                         # process one job and exit
python MVP/worker.py --recover                      # requeue stale 'running' jobs and exit
python MVP/compare_coaching_models.py [summary.json]  # benchmark local Ollama coaching models
```

There is no Python `pyproject.toml`, `requirements.txt`, or virtualenv config checked in. Dependencies are installed ad-hoc in the developer's environment. Known runtime deps: `supabase`, `python-dotenv`, `boto3`, `requests`, `opencv-python`, `mediapipe`, `numpy`, plus Apple `Vision`-framework bindings when `--pose-engine vision` is used.

### Env files (never commit)
- `MVP/web/.env.local` — Next.js side: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `R2_*`.
- `MVP/.env.worker` — Worker side: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `R2_*`, `LMSTUDIO_*`.

`.env.worker.example` and `.env.local.example` are the canonical templates. The root `.gitignore` blocks `.env*` except `*.example`.

## Architecture

### End-to-end job lifecycle
The system is a **3-process pipeline** coordinated through one Supabase `jobs` row whose `status` advances `created → queued → running → done|error`. State transitions:

1. **Browser → `/api/jobs/create`**: User picks a file. Route inserts a `jobs` row with status `created`, computes `storagePath = <user_id>/<job_id>/<safe_filename>`, and starts an R2 multipart upload session. Returns `{ jobId, uploadId, partSizeBytes, totalParts }`.
2. **Browser → R2 (direct)**: Browser fetches presigned PUT URLs for each part from `/api/jobs/upload-multipart`, uploads in parallel with retries, and posts the part ETags back to complete the multipart.
3. **Browser → `/api/jobs/mark-uploaded`**: Verifies the R2 object exists and has the expected size, then flips status to `queued`.
4. **Worker `_claim_job()`**: Polls for the oldest `queued` row, atomic-CAS-flips it to `running`, downloads the video from R2 (chunked with size verification), and shells out to `MVP/run.py`. The worker writes `progress_stage`, `progress_step`, `progress_note`, and `heartbeat_at` into `jobs.config` every minute so the UI can show live progress and the stale-job recovery (`STALE_THRESHOLD_S`, default 600s) can requeue dead workers.
5. **`run.py` → `TechniqueAnalysisRunner`**: Runs the full pipeline; writes a run directory under `technique-analysis/artifacts/runs/<timestamp>_<video_name>/` with `summary/summary.json`, `videos/overlay.mp4`, `metrics.csv`, and `mvp_summary.json` (compact handoff payload for the worker).
6. **Worker uploads artifacts**: `summary.json`, `metrics.csv`, `ai_coaching.json`, and per-turn "cool moment" JPEG frames → Supabase Storage bucket `artifacts`. The large `overlay.mp4` → R2 (`R2_ARTIFACTS_BUCKET`). Each artifact is recorded in the `artifacts` table with `meta.storage_provider` so the read path knows where to sign URLs from.
7. **Browser fetches recap**: `/api/jobs/[id]` (used by `/jobs/[id]` server component) joins the `artifacts` rows, signs short-lived R2 download URLs for R2-stored artifacts, and resolves Supabase Storage URLs for the rest.

### Storage split (Supabase + R2)
Originally everything lived in Supabase Storage; the repo has since migrated source videos and overlay artifacts to **Cloudflare R2** while keeping JSON/CSV/photo artifacts in Supabase Storage. The split is encoded in `jobs.config.video_storage_provider` (`'r2' | 'supabase'`) and `artifacts.meta.storage_provider`. Both worker (`_download_video_bytes`, `_upload_file_to_r2`) and web (`lib/r2.ts`, `lib/server-job-data.ts`) branch on this. Older jobs without the flag fall back to Supabase Storage — preserve this fallback when changing storage code. See `MVP/r2-video-upload-migration.md` for the original migration notes (including the required R2 CORS config — multipart uploads from the browser will fail silently with "Failed to fetch" if CORS is wrong for the exact origin/port).

### Auth model
Supabase Auth with `@supabase/ssr`. `middleware.ts` enforces:
- Anonymous (guest) users can `/upload` and view their own `/jobs/[id]` recap, but **not** `/jobs` (list) or `/profile` (those redirect to `/upload`).
- Signed-in non-anonymous users are redirected away from `/login` and `/signup` (but guests can visit `/signup` to upgrade their account).
- Public routes: `/`, `/login`, `/signup`, `/sample-analysis`, anything under `/api/`.

RLS policies (see `supabase/migrations/001_initial.sql`) only grant `select` on own rows. **All writes go through server routes using the service-role key** (`createServiceClient()` in `lib/supabase/server.ts`), so don't try to add client-side inserts/updates — they will be silently denied.

### Path conventions
- Web app uses `@/*` TS path alias mapped to `MVP/web/*` (see `tsconfig.json`).
- Python: `MVP/run.py` and `MVP/worker.py` add `technique-analysis/src/` to `sys.path` at runtime. There is no installed `technique_analysis` package — never `pip install -e` it; imports work because of the path insertion.
- Cool-moment frames, run directories, and artifact remote paths follow `jobs/<job_id>/...` naming. Match this when adding new artifact kinds.

### Technique analysis pipeline (`technique-analysis/src/technique_analysis/`)
Entry point: `free_ski.pipeline.orchestrator.TechniqueAnalysisRunner.run(video_path)`. Stages, in order:
1. `common.datasets.video_io.iter_frames` + `recommend_config` — frame loader with auto-downsampling.
2. `common.pose.extractor.PoseExtractor` (MediaPipe) **or** `common.pose.vision_extractor.VisionPoseExtractor` (Apple Vision) — selected by `TechniqueRunConfig.pose_engine`.
3. `common.pose.smoother.LandmarkSmoother` + gap-filling — Kalman smoothing with confidence decay across short detection gaps.
4. `common.metrics.frame_metrics.compute_frame_metrics` → per-frame angles/asymmetries → `compute_frame_score` adds composite `overall_score` + `movement_quality` label.
5. `common.turns.segmenter.segment_turns` → turn boundaries → `compute_turn_quality` fills `quality_score`, `smoothness_score`, `peak_lateral_shift`, `amplitude` per turn. Tracking segments (`TrackingSegment`) demarcate continuous athlete epochs when the tracker re-acquires after gaps.
6. `common.coaching.rules.generate_coaching_tips` — rule-based tips (LLM coaching from `lmstudio_coaching.py` is layered on top later, in the worker).
7. `common.rendering.overlay.render_overlay_video` — annotated overlay MP4.

All data classes live in `common.contracts.models` (single source of truth for the Python ↔ JSON ↔ TypeScript contract). The TypeScript mirror is in `MVP/web/lib/analysis-summary.ts` — if you change a `TurnSummary` / `CoachingTip` / `TechniqueRunSummary` field in Python, update the TS interfaces too, or the web recap will silently drop the data.

### Drill library — keep in sync
`DRILLS` is duplicated in three places: `MVP/lmstudio_coaching.py`, `MVP/claude_coaching.py`, `MVP/gemini_coaching.py`, and `MVP/web/lib/drills.ts`. When adding a drill, update all four (the coaching LLMs return `recommended_drill_id` and the web app looks it up).

### i18n
`MVP/web/lib/i18n.ts` holds an inline EN/ZH dictionary (`Lang = 'en' | 'zh'`). Server side reads `LANGUAGE_COOKIE = 'site_lang'` via `lib/i18n-server.ts`. The user's preferred language is also stored on the job (`jobs.config.preferred_language`) and passed to the coaching LLM so it generates in the right language.

## Design system
The visual language is documented in `ski-design-philosophy.md` ("Carved Arc"). Key invariants: cold sparse palette (pre-dawn granite / compacted-powder white) with warmth only as a rare accent; thin technical typography; coordinate-style data labels; diagonal composition energy. The reference moodboard image is `ski-carved-arc.png` at the repo root.

## Gotchas

- `yolov8n.pt` lives both at the repo root and in `MVP/` (~6.5 MB each). Both are gitignored (`*.pt`) but checked into the working tree.
- `technique-analysis/artifacts/` is gitignored — every `run.py` invocation produces a new dated run directory there. Don't write code that assumes a specific run is present.
- The worker shells out to `run.py` as a subprocess (rather than importing it) so a pipeline crash cannot kill the worker loop. Preserve this isolation when refactoring.
- `_extract_cool_moment_photos` in `worker.py` re-opens the source video with OpenCV after analysis — it depends on `full_summary.turns[*].start_s/end_s` and silently skips if `cv2` isn't installed.
- The `/api/jobs/create` route deletes the `jobs` row if the R2 multipart-upload-init fails, so the DB never holds orphaned rows pointing at non-existent uploads. Keep this cleanup if you refactor the route.
- LM Studio coaching failure is **non-fatal**: the worker logs a warning and continues without `ai_coaching.json` (`da8d08d8 Allow worker jobs to continue without AI coaching`). Don't tighten this to `raise`.
