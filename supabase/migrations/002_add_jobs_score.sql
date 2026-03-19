-- Persist technique score on completed jobs for progress dashboards.
ALTER TABLE public.jobs
  ADD COLUMN IF NOT EXISTS score real;

COMMENT ON COLUMN public.jobs.score IS 'Technique score (0-100) computed from summary_json, persisted for efficient progress queries';
