import { NextRequest, NextResponse } from 'next/server'
import { buildTechniqueDashboard, type TechniqueRunSummary } from '@/lib/analysis-summary'
import { createClient, createServiceClient } from '@/lib/supabase/server'

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

  // Fetch job
  const { data: job, error: jobError } = await service
    .from('jobs')
    .select('*')
    .eq('id', jobId)
    .single()

  if (jobError || !job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  // Ownership check
  if (job.user_id !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  // Fetch artifacts
  const { data: artifacts } = await service
    .from('artifacts')
    .select('*')
    .eq('job_id', jobId)
    .order('created_at')

  // Generate 1-hour signed download URLs for each artifact
  const artifactsWithUrls = await Promise.all(
    (artifacts ?? []).map(async (artifact) => {
      const { data } = await service.storage
        .from('artifacts')
        .createSignedUrl(artifact.object_path, 3600)
      return { ...artifact, url: data?.signedUrl ?? '' }
    })
  )

  let summary: TechniqueRunSummary | null = null
  const summaryArtifact = (artifacts ?? []).find((artifact) => artifact.kind === 'summary_json')

  if (summaryArtifact) {
    const { data: summaryFile } = await service.storage
      .from('artifacts')
      .download(summaryArtifact.object_path)

    if (summaryFile) {
      try {
        summary = JSON.parse(await summaryFile.text()) as TechniqueRunSummary
      } catch (error) {
        console.error('summary parse error:', error)
      }
    }
  }

  // Persist score if summary exists and job.score is not yet set
  if (summary && job.score == null && job.status === 'done') {
    const dashboard = buildTechniqueDashboard(summary)
    const computedScore = dashboard.overview.overallScore
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
      .select('score')
      .eq('user_id', user.id)
      .eq('status', 'done')
      .lt('created_at', job.created_at)
      .not('score', 'is', null)
      .order('created_at', { ascending: false })
      .limit(1)

    if (prevJobs?.length && prevJobs[0].score != null) {
      previousScore = prevJobs[0].score
    }
  }

  return NextResponse.json({ job, artifacts: artifactsWithUrls, summary, previousScore })
}
