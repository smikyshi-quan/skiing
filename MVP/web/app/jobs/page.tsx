import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'
import { Job, JobStatus } from '@/lib/types'

const STATUS_PILL: Record<JobStatus, string> = {
  created:  'bg-gray-100 text-gray-600',
  uploaded: 'bg-blue-50  text-blue-700',
  queued:   'bg-yellow-50 text-yellow-700',
  running:  'bg-blue-100 text-blue-800',
  done:     'bg-green-50 text-green-700',
  error:    'bg-red-50   text-red-700',
}

export const dynamic = 'force-dynamic'

export default async function JobsPage() {
  const supabase = createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) redirect('/login')

  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .order('created_at', { ascending: false })

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Analysis jobs</h1>
        <Link
          href="/upload"
          className="bg-blue-600 text-white rounded-md px-4 py-2 text-sm font-medium hover:bg-blue-700 transition-colors"
        >
          New analysis
        </Link>
      </div>

      {!jobs?.length ? (
        <div className="text-center py-20 text-gray-400">
          <p className="mb-3">No analyses yet.</p>
          <Link href="/upload" className="text-blue-600 hover:underline text-sm">
            Upload your first video
          </Link>
        </div>
      ) : (
        <ul className="space-y-2.5">
          {jobs.map((job: Job) => (
            <li key={job.id}>
              <Link
                href={`/jobs/${job.id}`}
                className="flex items-center justify-between px-4 py-3.5 bg-white border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
              >
                <div className="min-w-0">
                  <p className="text-xs font-mono text-gray-400 truncate">
                    {job.video_object_path?.split('/').pop() ?? job.id.slice(0, 8)}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(job.created_at).toLocaleString()}
                  </p>
                </div>
                <span
                  className={`ml-4 shrink-0 text-xs font-medium px-2.5 py-1 rounded-full ${
                    STATUS_PILL[job.status as JobStatus]
                  }`}
                >
                  {job.status}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
