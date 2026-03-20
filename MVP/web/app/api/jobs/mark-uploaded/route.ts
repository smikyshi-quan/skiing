import { NextRequest, NextResponse } from 'next/server'
import { createClient, createServiceClient } from '@/lib/supabase/server'

export async function POST(req: NextRequest) {
  const supabase = createClient()
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser()

  if (authError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await req.json().catch(() => ({}))
  const { jobId } = body
  if (!jobId || typeof jobId !== 'string') {
    return NextResponse.json({ error: '`jobId` is required' }, { status: 400 })
  }

  const service = createServiceClient()

  // Confirm ownership and current state before mutating
  const { data: job } = await service
    .from('jobs')
    .select('id, user_id, status, video_object_path')
    .eq('id', jobId)
    .single()

  if (!job || job.user_id !== user.id) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  // Only transition from 'created' to 'queued'
  if (job.status !== 'created') {
    return NextResponse.json(
      { error: `Job is in '${job.status}' state, expected 'created'` },
      { status: 409 }
    )
  }

  // Verify the video object actually exists in storage
  if (job.video_object_path) {
    const { data: objects, error: listError } = await service.storage
      .from('videos')
      .list(job.video_object_path.split('/').slice(0, -1).join('/'), {
        search: job.video_object_path.split('/').pop(),
        limit: 1,
      })

    if (listError || !objects?.length) {
      return NextResponse.json(
        { error: 'Video file not found in storage. Upload may have failed.' },
        { status: 400 }
      )
    }
  }

  await service
    .from('jobs')
    .update({ status: 'queued' })
    .eq('id', jobId)

  return NextResponse.json({ ok: true })
}
