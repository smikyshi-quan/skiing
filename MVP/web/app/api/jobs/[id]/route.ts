import { NextRequest, NextResponse } from 'next/server'
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

  return NextResponse.json({ job, artifacts: artifactsWithUrls })
}
