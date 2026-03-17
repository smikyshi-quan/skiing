import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'
import { Job, JobStatus } from '@/lib/types'

export const dynamic = 'force-dynamic'

const STATUS_CONFIG: Record<JobStatus, { label: string; dot: string; pill: string }> = {
  created:  { label: 'Created',   dot: 'rgba(255,255,255,0.25)', pill: 'rgba(255,255,255,0.07)' },
  uploaded: { label: 'Uploaded',  dot: '#4F8EFF',                pill: 'rgba(79,142,255,0.12)' },
  queued:   { label: 'Queued',    dot: '#F59E0B',                pill: 'rgba(245,158,11,0.12)' },
  running:  { label: 'Analysing', dot: '#4F8EFF',                pill: 'rgba(79,142,255,0.14)' },
  done:     { label: 'Done',      dot: '#22D07A',                pill: 'rgba(34,208,122,0.12)' },
  error:    { label: 'Error',     dot: '#F87171',                pill: 'rgba(248,113,113,0.12)' },
}

const STATUS_TEXT: Record<JobStatus, string> = {
  created:  'rgba(255,255,255,0.5)',
  uploaded: '#93C5FD',
  queued:   '#FCD34D',
  running:  '#93C5FD',
  done:     '#6EE7B7',
  error:    '#FCA5A5',
}

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
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">My Runs</h1>
          <p className="text-sm mt-1" style={{ color: 'rgba(255,255,255,0.35)' }}>
            {jobs?.length ? `${jobs.length} analysis${jobs.length !== 1 ? 'es' : ''}` : 'No analyses yet'}
          </p>
        </div>
        <Link
          href="/upload"
          className="btn-primary flex items-center gap-2"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14"/>
          </svg>
          New run
        </Link>
      </div>

      {/* Empty state */}
      {!jobs?.length ? (
        <div
          className="rounded-2xl p-16 text-center"
          style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}
        >
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
            style={{ background: 'rgba(255,255,255,0.05)' }}
          >
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 8v4M12 16h.01"/>
            </svg>
          </div>
          <p className="text-sm font-medium text-white mb-1">No analyses yet</p>
          <p className="text-xs mb-5" style={{ color: 'rgba(255,255,255,0.3)' }}>
            Upload a ski video to get started
          </p>
          <Link
            href="/upload"
            className="btn-primary inline-flex items-center gap-2"
          >
            Upload your first video
          </Link>
        </div>
      ) : (
        <ul className="space-y-3">
          {jobs.map((job: Job) => {
            const cfg = STATUS_CONFIG[job.status as JobStatus]
            const filename =
              String(job.config?.original_filename ?? '') ||
              job.video_object_path?.split('/').pop() ||
              job.id.slice(0, 8)
            const isRunning = job.status === 'running' || job.status === 'queued'

            return (
              <li key={job.id}>
                <Link
                  href={`/jobs/${job.id}`}
                  className="card card-hover flex items-center gap-4 px-5 py-4 group"
                  style={{ display: 'flex' }}
                >
                  {/* Status dot / icon */}
                  <div
                    className="w-9 h-9 rounded-xl shrink-0 flex items-center justify-center"
                    style={{ background: cfg.pill }}
                  >
                    {isRunning ? (
                      <div
                        className="w-2.5 h-2.5 rounded-full animate-pulse"
                        style={{ background: cfg.dot }}
                      />
                    ) : (
                      <div className="w-2.5 h-2.5 rounded-full" style={{ background: cfg.dot }} />
                    )}
                  </div>

                  {/* Content */}
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-white truncate">{filename}</p>
                    <p className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
                      {new Date(job.created_at).toLocaleString()}
                    </p>
                  </div>

                  {/* Status badge */}
                  <span
                    className="shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full"
                    style={{ background: cfg.pill, color: STATUS_TEXT[job.status as JobStatus] }}
                  >
                    {cfg.label}
                  </span>

                  {/* Arrow */}
                  <svg
                    width="14" height="14"
                    viewBox="0 0 24 24" fill="none"
                    stroke="rgba(255,255,255,0.2)"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    className="shrink-0 transition-all group-hover:translate-x-0.5"
                    style={{ transition: 'transform 150ms, stroke 150ms' }}
                  >
                    <path d="M9 18l6-6-6-6"/>
                  </svg>
                </Link>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
