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

  // Confirm ownership before mutating
  const { data: job } = await service
    .from('jobs')
    .select('id, user_id, status')
    .eq('id', jobId)
    .single()

  if (!job || job.user_id !== user.id) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  await service
    .from('jobs')
    .update({ status: 'queued' })
    .eq('id', jobId)

  return NextResponse.json({ ok: true })
}
