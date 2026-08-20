import 'server-only'

import { createArtifactDownloadUrl, getDefaultR2ArtifactsBucket } from '@/lib/r2'
import { createServiceClient } from '@/lib/supabase/server'
import { buildTechniqueDashboard, scoreCountsForProgress, type AiCoaching, type TechniqueRunSummary } from '@/lib/analysis-summary'
import type { Job } from '@/lib/types'

type ServiceClient = ReturnType<typeof createServiceClient>

interface SummaryArtifactRow {
  job_id: string
  object_path: string
}

interface CoachingArtifactRow {
  job_id: string
  kind: string
  object_path: string
}

interface PreviewArtifactRow {
  job_id: string
  kind: string
  object_path: string
  meta?: Record<string, unknown>
}

export const CREATED_UPLOAD_STALE_MINUTES = 15
export const ANALYSIS_STALE_MINUTES = 10

export type JobPresentationTone = 'neutral' | 'accent' | 'warning' | 'success' | 'danger'

export interface JobPresentation {
  state: 'preparing' | 'upload_incomplete' | 'processing' | 'needs_attention' | 'ready' | 'failed'
  label: string
  helper: string
  tone: JobPresentationTone
  dot: string
  pill: string
  canRetry: boolean
  retryable: boolean
  actionLabel: string | null
  isUploadIncomplete: boolean
  isStalled: boolean
}

function parseDate(value: unknown): Date | null {
  if (typeof value !== 'string' || !value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

function minutesSince(date: Date | null, now: Date) {
  if (!date) return null
  return (now.getTime() - date.getTime()) / 60000
}

function jobConfig(job: Job) {
  return (job.config ?? {}) as Record<string, unknown>
}

function jobProgressText(job: Job) {
  const config = jobConfig(job)
  const progressStage = typeof config.progress_stage === 'string' && config.progress_stage.trim()
    ? config.progress_stage.trim()
    : null
  const progressNote = typeof config.progress_note === 'string' && config.progress_note.trim()
    ? config.progress_note.trim()
    : null

  return progressStage ?? progressNote
}

export function jobHasUpload(job: Job) {
  return typeof job.video_object_path === 'string' && job.video_object_path.length > 0
}

export function jobFreshnessMinutes(job: Job, now = new Date()) {
  const config = jobConfig(job)
  const heartbeatAt = parseDate(config.heartbeat_at)
  const updatedAt = parseDate(job.updated_at)
  const base = heartbeatAt ?? updatedAt
  return minutesSince(base, now)
}

export function jobIsRetryable(job: Job, now = new Date()) {
  if (!jobHasUpload(job)) return false
  if (job.status === 'error') return true

  const freshnessMinutes = jobFreshnessMinutes(job, now)
  if (freshnessMinutes == null) return false

  return (job.status === 'uploaded' || job.status === 'queued' || job.status === 'running')
    && freshnessMinutes >= ANALYSIS_STALE_MINUTES
}

export function resolveJobPresentation(job: Job, now = new Date()): JobPresentation {
  const ageMinutes = minutesSince(parseDate(job.created_at), now)
  const progressText = jobProgressText(job)
  const hasUpload = jobHasUpload(job)
  const staleAnalysis = jobIsRetryable(job, now)
  const createdTooOld = job.status === 'created'
    && ageMinutes != null
    && ageMinutes >= CREATED_UPLOAD_STALE_MINUTES

  if (createdTooOld) {
    return {
      state: 'upload_incomplete',
      label: 'Upload incomplete',
      helper: 'This upload never finished. Start a new upload to try again.',
      tone: 'warning',
      dot: 'var(--gold)',
      pill: 'var(--gold-dim)',
      canRetry: false,
      retryable: false,
      actionLabel: 'Upload again',
      isUploadIncomplete: true,
      isStalled: false,
    }
  }

  if (job.status === 'error') {
    return {
      state: 'failed',
      label: 'Analysis failed',
      helper: job.error ?? 'The analysis hit an error. Retry with the same video if it is still available.',
      tone: 'danger',
      dot: 'var(--danger)',
      pill: 'var(--danger-dim)',
      canRetry: hasUpload,
      retryable: hasUpload,
      actionLabel: hasUpload ? 'Retry analysis' : 'Upload again',
      isUploadIncomplete: false,
      isStalled: false,
    }
  }

  if (staleAnalysis) {
    return {
      state: 'needs_attention',
      label: 'Needs attention',
      helper: 'This analysis has not updated recently. Retry it if the uploaded video is still in storage.',
      tone: 'warning',
      dot: 'var(--gold)',
      pill: 'var(--gold-dim)',
      canRetry: true,
      retryable: true,
      actionLabel: 'Retry analysis',
      isUploadIncomplete: false,
      isStalled: true,
    }
  }

  if (job.status === 'done') {
    return {
      state: 'ready',
      label: 'Recap ready',
      helper: 'Your feedback is ready to review.',
      tone: 'success',
      dot: 'var(--success)',
      pill: 'var(--success-dim)',
      canRetry: false,
      retryable: false,
      actionLabel: null,
      isUploadIncomplete: false,
      isStalled: false,
    }
  }

  if (job.status === 'created') {
    return {
      state: 'preparing',
      label: 'Preparing upload',
      helper: 'We are waiting for the video upload to finish.',
      tone: 'neutral',
      dot: 'var(--ink-muted)',
      pill: 'rgba(0,0,0,0.04)',
      canRetry: false,
      retryable: false,
      actionLabel: null,
      isUploadIncomplete: false,
      isStalled: false,
    }
  }

  const labelByStatus: Record<Job['status'], string> = {
    created: 'Preparing upload',
    uploaded: 'Upload complete',
    queued: 'Queued',
    running: 'Analyzing',
    done: 'Recap ready',
    error: 'Analysis failed',
  }

  const helperByStatus: Record<Job['status'], string> = {
    created: 'We are waiting for the video upload to finish.',
    uploaded: progressText ?? 'Your video is ready. Analysis will begin shortly.',
    queued: progressText ?? 'We are getting your analysis started.',
    running: progressText ?? 'We are reviewing your technique and preparing your recap.',
    done: 'Your feedback is ready to review.',
    error: job.error ?? 'The analysis hit an error. Retry with the same video if it is still available.',
  }

  const toneByStatus: Record<Job['status'], JobPresentationTone> = {
    created: 'neutral',
    uploaded: 'accent',
    queued: 'warning',
    running: 'accent',
    done: 'success',
    error: 'danger',
  }

  const dotByStatus: Record<Job['status'], string> = {
    created: 'var(--ink-muted)',
    uploaded: 'var(--accent)',
    queued: 'var(--gold)',
    running: 'var(--accent)',
    done: 'var(--success)',
    error: 'var(--danger)',
  }

  const pillByStatus: Record<Job['status'], string> = {
    created: 'rgba(0,0,0,0.04)',
    uploaded: 'var(--accent-dim)',
    queued: 'var(--gold-dim)',
    running: 'var(--accent-dim)',
    done: 'var(--success-dim)',
    error: 'var(--danger-dim)',
  }

  return {
    state: job.status === 'uploaded' || job.status === 'queued' || job.status === 'running'
      ? 'processing'
      : job.status === 'done'
        ? 'ready'
        : job.status === 'error'
          ? 'failed'
          : 'preparing',
    label: labelByStatus[job.status],
    helper: helperByStatus[job.status],
    tone: toneByStatus[job.status],
    dot: dotByStatus[job.status],
    pill: pillByStatus[job.status],
    canRetry: false,
    retryable: false,
    actionLabel: null,
    isUploadIncomplete: false,
    isStalled: false,
  }
}

function artifactStorageProvider(artifact: { meta?: Record<string, unknown> }) {
  return artifact.meta?.storage_provider === 'r2' ? 'r2' : 'supabase'
}

function artifactStorageBucket(artifact: { meta?: Record<string, unknown> }) {
  const metaBucket = artifact.meta?.storage_bucket
  return typeof metaBucket === 'string' && metaBucket ? metaBucket : getDefaultR2ArtifactsBucket()
}

export function computeSummaryScore(summary: TechniqueRunSummary): number | null {
  const score = buildTechniqueDashboard(summary).overview.overallScore
  return Number.isFinite(score) ? score : null
}

export async function loadSummaryFromObjectPath(
  service: ServiceClient,
  objectPath: string,
): Promise<TechniqueRunSummary | null> {
  const { data: file } = await service.storage
    .from('artifacts')
    .download(objectPath)

  if (!file) return null

  try {
    return JSON.parse(await file.text()) as TechniqueRunSummary
  } catch {
    return null
  }
}

export async function loadSummariesForJobIds(
  service: ServiceClient,
  jobIds: string[],
): Promise<Map<string, TechniqueRunSummary>> {
  const summaryByJob = new Map<string, TechniqueRunSummary>()
  if (!jobIds.length) return summaryByJob

  const { data: artifacts } = await service
    .from('artifacts')
    .select('job_id, object_path')
    .in('job_id', jobIds)
    .eq('kind', 'summary_json')

  const rows = (artifacts ?? []) as SummaryArtifactRow[]
  await Promise.all(rows.map(async (artifact) => {
    const summary = await loadSummaryFromObjectPath(service, artifact.object_path)
    if (summary) {
      summaryByJob.set(artifact.job_id, summary)
    }
  }))

  return summaryByJob
}

export async function loadAiCoachingForJobIds(
  service: ServiceClient,
  jobIds: string[],
): Promise<Map<string, AiCoaching>> {
  const coachingByJob = new Map<string, AiCoaching>()
  if (!jobIds.length) return coachingByJob

  const { data: artifacts } = await service
    .from('artifacts')
    .select('job_id, kind, object_path')
    .in('job_id', jobIds)
    .in('kind', ['ai_coaching', 'claude_coaching', 'gemini_coaching'])
    .order('created_at', { ascending: false })

  const artifactByJob = new Map<string, CoachingArtifactRow>()
  for (const artifact of (artifacts ?? []) as CoachingArtifactRow[]) {
    const current = artifactByJob.get(artifact.job_id)
    if (!current || (current.kind !== 'ai_coaching' && artifact.kind === 'ai_coaching')) {
      artifactByJob.set(artifact.job_id, artifact)
    }
  }

  await Promise.all(Array.from(artifactByJob.values()).map(async (artifact) => {
    const { data: file } = await service.storage
      .from('artifacts')
      .download(artifact.object_path)

    if (!file) return

    try {
      coachingByJob.set(artifact.job_id, JSON.parse(await file.text()) as AiCoaching)
    } catch {
      // Ignore malformed optional judge artifacts; the UI will show unavailable.
    }
  }))

  return coachingByJob
}

export function jobScoreCountsForProgress(
  job: Pick<Job, 'score'>,
  summary: TechniqueRunSummary | null | undefined,
): boolean {
  if (job.score == null) return true
  return scoreCountsForProgress(summary)
}

export async function loadPreviewUrlsForJobIds(
  service: ServiceClient,
  jobIds: string[],
): Promise<Map<string, string>> {
  const previewUrlByJob = new Map<string, string>()
  if (!jobIds.length) return previewUrlByJob

  const { data: artifacts } = await service
    .from('artifacts')
    .select('job_id, kind, object_path, meta')
    .in('job_id', jobIds)
    .in('kind', ['cool_moment_photo', 'peak_pressure_frame', 'peak_pressure_frame_enhanced'])
    .order('created_at')

  const previewByJob = new Map<string, PreviewArtifactRow>()
  for (const artifact of (artifacts ?? []) as PreviewArtifactRow[]) {
    const current = previewByJob.get(artifact.job_id)
    if (!current) {
      previewByJob.set(artifact.job_id, artifact)
      continue
    }
    if (current.kind !== 'cool_moment_photo' && artifact.kind === 'cool_moment_photo') {
      previewByJob.set(artifact.job_id, artifact)
    }
  }

  await Promise.all(Array.from(previewByJob.values()).map(async (artifact) => {
    if (artifactStorageProvider(artifact) === 'r2') {
      previewUrlByJob.set(
        artifact.job_id,
        await createArtifactDownloadUrl(artifact.object_path, artifactStorageBucket(artifact)),
      )
      return
    }

    const { data } = await service.storage
      .from('artifacts')
      .createSignedUrl(artifact.object_path, 3600)

    if (data?.signedUrl) {
      previewUrlByJob.set(artifact.job_id, data.signedUrl)
    }
  }))

  return previewUrlByJob
}

export async function backfillMissingScores(
  service: ServiceClient,
  jobs: Job[],
): Promise<Map<string, number>> {
  const missingScoreJobs = jobs.filter((job) => job.status === 'done' && job.score == null)
  const derivedScores = new Map<string, number>()
  if (!missingScoreJobs.length) return derivedScores

  const summaries = await loadSummariesForJobIds(service, missingScoreJobs.map((job) => job.id))

  await Promise.all(missingScoreJobs.map(async (job) => {
    const summary = summaries.get(job.id)
    if (!summary) return

    const score = computeSummaryScore(summary)
    if (score == null) return

    derivedScores.set(job.id, score)
    job.score = score

    await service
      .from('jobs')
      .update({ score })
      .eq('id', job.id)
  }))

  return derivedScores
}
