import { NextRequest, NextResponse } from 'next/server'
import { createClient, createServiceClient } from '@/lib/supabase/server'

export async function POST(req: NextRequest) {
  // 1. Verify the caller is authenticated
  const supabase = createClient()
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser()

  if (authError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await req.json().catch(() => ({}))
  const { filename } = body
  if (!filename || typeof filename !== 'string') {
    return NextResponse.json({ error: '`filename` is required' }, { status: 400 })
  }

  const service = createServiceClient()

  // 2. Insert job row
  const { data: job, error: jobError } = await service
    .from('jobs')
    .insert({ user_id: user.id, status: 'created' })
    .select()
    .single()

  if (jobError || !job) {
    console.error('jobs insert error:', jobError)
    return NextResponse.json(
      { error: jobError?.message ?? 'Failed to create job' },
      { status: 500 }
    )
  }

  // 3. Generate signed upload URL  →  videos/<user_id>/<job_id>/<filename>
  const storagePath = `${user.id}/${job.id}/${filename}`
  const { data: signed, error: signedError } = await service.storage
    .from('videos')
    .createSignedUploadUrl(storagePath)

  if (signedError || !signed) {
    console.error('signed upload URL error:', signedError)
    return NextResponse.json(
      { error: signedError?.message ?? 'Failed to create upload URL' },
      { status: 500 }
    )
  }

  // 4. Persist the video path on the job so the worker can find it
  await service
    .from('jobs')
    .update({ video_object_path: storagePath })
    .eq('id', job.id)

  return NextResponse.json({
    jobId: job.id,
    path: signed.path,
    token: signed.token,
  })
}
