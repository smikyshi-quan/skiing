import { NextRequest, NextResponse } from 'next/server'
import { scoreCountsForProgress, type TechniqueRunSummary, type AiCoaching } from '@/lib/analysis-summary'
import { createClient, createServiceClient } from '@/lib/supabase/server'
import { createArtifactDownloadUrl, getDefaultR2ArtifactsBucket } from '@/lib/r2'
import {
  computeSummaryScore,
  jobHasUpload,
  jobIsRetryable,
  loadSummariesForJobIds,
  loadSummaryFromObjectPath,
  resolveJobPresentation,
} from '@/lib/server-job-data'
import { displayNameFromFilename } from '@/lib/job-ui'
import { LANGUAGE_COOKIE, normalizeLang } from '@/lib/i18n'

function artifactStorageProvider(artifact: { meta?: Record<string, unknown> }) {
  return artifact.meta?.storage_provider === 'r2' ? 'r2' : 'supabase'
}

function artifactStorageBucket(artifact: { meta?: Record<string, unknown> }) {
  const metaBucket = artifact.meta?.storage_bucket
  return typeof metaBucket === 'string' && metaBucket ? metaBucket : getDefaultR2ArtifactsBucket()
}

function normalizeEditableText(value: unknown, maxLength: number) {
  if (value == null) return null
  if (typeof value !== 'string') return undefined

  const normalized = value.trim().replace(/\s+/g, ' ')
  if (!normalized) return null
  return normalized.slice(0, maxLength)
}

function normalizeExpectedSize(value: unknown) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return null
  }

  return Math.floor(value)
}

async function videoObjectExists(
  service: ReturnType<typeof createServiceClient>,
  job: { video_object_path: string | null; config: Record<string, unknown> },
) {
  if (!job.video_object_path) return false

  const provider = job.config.video_storage_provider === 'r2' ? 'r2' : 'supabase'
  const expectedSizeBytes = normalizeExpectedSize(job.config.video_file_size_bytes)

  if (provider === 'r2') {
    const { getVideoObjectMetadata } = await import('@/lib/r2')
    const metadata = await getVideoObjectMetadata(job.video_object_path)
    if (!metadata.exists) return false

    if (expectedSizeBytes != null && metadata.sizeBytes !== expectedSizeBytes) {
      return false
    }

    return true
  }

  const { data: objects, error: listError } = await service.storage
    .from('videos')
    .list(job.video_object_path.split('/').slice(0, -1).join('/'), {
      search: job.video_object_path.split('/').pop(),
      limit: 1,
    })

  if (listError || !objects?.length) {
    return false
  }

  return true
}

function clearProgressConfig(config: Record<string, unknown>, sourceStatus: string) {
  const nextConfig = { ...config }
  nextConfig.retry_requested_at = new Date().toISOString()
  nextConfig.retry_requested_from = sourceStatus
  delete nextConfig.progress_note
  delete nextConfig.heartbeat_at
  delete nextConfig.progress_step
  delete nextConfig.progress_total
  delete nextConfig.progress_stage
  return nextConfig
}

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  const supabase = createClient()
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser()

  if (authError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const service = createServiceClient()
  const jobId = params.id

  // Fetch job using the session client — RLS scopes to the authenticated user
  const { data: job, error: jobError } = await supabase
    .from('jobs')
    .select('*')
    .eq('id', jobId)
    .single()

  if (jobError || !job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  // Fetch artifacts — use service client for storage signing
  const { data: artifacts } = await service
    .from('artifacts')
    .select('*')
    .eq('job_id', jobId)
    .order('created_at')

  // Generate 1-hour signed download URLs for each artifact
  const artifactsWithUrls = await Promise.all(
    (artifacts ?? []).map(async (artifact) => {
      if (artifactStorageProvider(artifact as { meta?: Record<string, unknown> }) === 'r2') {
        const url = await createArtifactDownloadUrl(
          artifact.object_path,
          artifactStorageBucket(artifact as { meta?: Record<string, unknown> })
        )
        return { ...artifact, url }
      }

      const { data } = await service.storage
        .from('artifacts')
        .createSignedUrl(artifact.object_path, 3600)
      return { ...artifact, url: data?.signedUrl ?? '' }
    })
  )

  let summary: TechniqueRunSummary | null = null
  const summaryArtifact = (artifacts ?? []).find((artifact) => artifact.kind === 'summary_json')

  if (summaryArtifact) {
    summary = await loadSummaryFromObjectPath(service, summaryArtifact.object_path)
  }

  // Fetch current AI coaching JSON if available, with legacy fallback.
  let aiCoaching: AiCoaching | null = null
  const coachingArtifact = (artifacts ?? []).find((artifact) => artifact.kind === 'ai_coaching')
    ?? (artifacts ?? []).find((artifact) => artifact.kind === 'claude_coaching')
    ?? (artifacts ?? []).find((artifact) => artifact.kind === 'gemini_coaching')
  if (coachingArtifact) {
    const { data: coachingFile } = await service.storage
      .from('artifacts')
      .download(coachingArtifact.object_path)
    if (coachingFile) {
      try {
        aiCoaching = JSON.parse(await coachingFile.text()) as AiCoaching
      } catch (error) {
        console.error('ai coaching parse error:', error)
      }
    }
  }

  // Persist score if summary exists and job.score is not yet set
  if (summary && job.score == null && job.status === 'done') {
    const computedScore = computeSummaryScore(summary)
    if (Number.isFinite(computedScore)) {
      await service
        .from('jobs')
        .update({ score: computedScore })
        .eq('id', jobId)
      job.score = computedScore
    }
  }

  // Find previous completed run's score for delta
  let previousScore: number | null = null
  if (job.status === 'done') {
    const { data: prevJobs } = await service
      .from('jobs')
      .select('id, score')
      .eq('user_id', user.id)
      .eq('status', 'done')
      .lt('created_at', job.created_at)
      .not('score', 'is', null)
      .order('created_at', { ascending: false })
      .limit(20)

    const previousRuns = (prevJobs ?? []) as Array<{ id: string; score: number | null }>
    const previousSummaryByJob = await loadSummariesForJobIds(service, previousRuns.map((prevJob) => prevJob.id))
    const previousProgressRun = previousRuns.find((prevJob) => {
      if (prevJob.score == null) return false
      const previousSummary = previousSummaryByJob.get(prevJob.id)
      return scoreCountsForProgress(previousSummary)
    })

    if (previousProgressRun?.score != null) {
      previousScore = previousProgressRun.score
    }
  }

  return NextResponse.json({
    job,
    artifacts: artifactsWithUrls,
    summary,
    previousScore,
    aiCoaching,
    presentation: resolveJobPresentation(job),
  })
}

export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const supabase = createClient()
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser()

  if (authError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await req.json().catch(() => ({}))
  if (body?.action !== 'retry') {
    return NextResponse.json({ error: 'Unsupported action' }, { status: 400 })
  }

  const service = createServiceClient()
  const jobId = params.id

  const { data: job, error: jobError } = await supabase
    .from('jobs')
    .select('*')
    .eq('id', jobId)
    .single()

  if (jobError || !job || job.user_id !== user.id) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  const presentation = resolveJobPresentation(job)
  if (presentation.state === 'upload_incomplete') {
    return NextResponse.json(
      { error: 'This upload never finished. Start a new upload instead of retrying this draft.' },
      { status: 409 }
    )
  }

  if (!jobIsRetryable(job) && job.status !== 'error') {
    return NextResponse.json(
      { error: 'Only stalled or failed runs can be retried.' },
      { status: 409 }
    )
  }

  if (!jobHasUpload(job)) {
    return NextResponse.json(
      { error: 'This run does not have an uploaded video to retry.' },
      { status: 409 }
    )
  }

  const storedVideoExists = await videoObjectExists(service, {
    video_object_path: job.video_object_path,
    config: (job.config ?? {}) as Record<string, unknown>,
  })

  if (!storedVideoExists) {
    return NextResponse.json(
      { error: 'The uploaded video could not be found in storage. Please upload it again.' },
      { status: 400 }
    )
  }

  const nextConfig = clearProgressConfig((job.config ?? {}) as Record<string, unknown>, job.status)
  nextConfig.preferred_language = normalizeLang(req.cookies.get(LANGUAGE_COOKIE)?.value)

  const { data: updatedJob, error: updateError } = await service
    .from('jobs')
    .update({
      status: 'queued',
      error: null,
      config: nextConfig,
      updated_at: new Date().toISOString(),
    })
    .eq('id', jobId)
    .select('*')
    .single()

  if (updateError || !updatedJob) {
    console.error('job retry update error:', updateError)
    return NextResponse.json({ error: 'Failed to retry this run.' }, { status: 500 })
  }

  return NextResponse.json({
    ok: true,
    job: updatedJob,
    presentation: resolveJobPresentation(updatedJob),
  })
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const supabase = createClient()
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser()

  if (authError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await req.json().catch(() => ({}))
  const displayNameInput = normalizeEditableText(body.displayName, 80)
  const userNoteInput = normalizeEditableText(body.userNote, 240)

  if (displayNameInput === undefined && userNoteInput === undefined) {
    return NextResponse.json({ error: 'No editable fields were provided' }, { status: 400 })
  }

  const service = createServiceClient()
  const jobId = params.id

  const { data: job, error: jobError } = await supabase
    .from('jobs')
    .select('id, user_id, config')
    .eq('id', jobId)
    .single()

  if (jobError || !job || job.user_id !== user.id) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  const nextConfig = { ...(job.config ?? {}) } as Record<string, unknown>

  if (displayNameInput !== undefined) {
    if (displayNameInput == null) {
      const originalFilename = typeof nextConfig.original_filename === 'string' ? nextConfig.original_filename : ''
      nextConfig.display_name = displayNameFromFilename(originalFilename || job.id.slice(0, 8))
    } else {
      nextConfig.display_name = displayNameInput
    }
  }

  if (userNoteInput !== undefined) {
    if (userNoteInput == null) {
      delete nextConfig.user_note
    } else {
      nextConfig.user_note = userNoteInput
    }
  }

  const { data: updatedJob, error: updateError } = await service
    .from('jobs')
    .update({ config: nextConfig })
    .eq('id', jobId)
    .select('*')
    .single()

  if (updateError || !updatedJob) {
    return NextResponse.json({ error: updateError?.message ?? 'Failed to update run metadata' }, { status: 500 })
  }

  return NextResponse.json({ job: updatedJob })
}
