import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'
import { Job, JobStatus } from '@/lib/types'
import { groupBySeason } from '@/lib/seasons'
import { scoreLabel } from '@/lib/analysis-summary'

export const dynamic = 'force-dynamic'

const STATUS_CONFIG: Record<JobStatus, { label: string; dot: string; pill: string }> = {
  created:  { label: 'Created',   dot: 'var(--ink-muted)',  pill: 'rgba(255,255,255,0.04)' },
  uploaded: { label: 'Uploaded',  dot: 'var(--accent)',     pill: 'var(--accent-dim)' },
  queued:   { label: 'Queued',    dot: 'var(--gold)',       pill: 'var(--gold-dim)' },
  running:  { label: 'Analysing', dot: 'var(--accent)',     pill: 'var(--accent-dim)' },
  done:     { label: 'Done',      dot: 'var(--success)',    pill: 'var(--success-dim)' },
  error:    { label: 'Error',     dot: 'var(--danger)',     pill: 'var(--danger-dim)' },
}

function levelBadgeClass(label: string) {
  switch (label) {
    case 'Focus': return 'level-badge level-badge--focus'
    case 'Building': return 'level-badge level-badge--building'
    case 'Good': return 'level-badge level-badge--good'
    case 'Dialed': return 'level-badge level-badge--dialed'
    default: return 'level-badge level-badge--building'
  }
}

export default async function ArchivePage() {
  const supabase = createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) redirect('/login')

  // RLS policy on `jobs` scopes results to the authenticated user
  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .order('created_at', { ascending: false })

  const runs = (jobs ?? []) as Job[]
  const completedRuns = runs.filter((job) => job.status === 'done')

  // Season grouping
  const seasonGroups = groupBySeason(runs)

  // Season summary stats
  const scoredRuns = completedRuns.filter((j) => j.score != null) as (Job & { score: number })[]
  const avgScore = scoredRuns.length
    ? Math.round(scoredRuns.reduce((sum, j) => sum + j.score, 0) / scoredRuns.length)
    : null

  return (
    <div className="space-y-6">
      <section className="surface-card p-8 lg:p-10">
        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <span className="eyebrow">Run archive</span>
            <h1 className="section-title mt-6">Every session, captured and ready to revisit.</h1>
            <p className="section-copy mt-4 max-w-xl">
              Your full history of uploaded runs, grouped by ski season. Tap into any recap to review technique scores, key moments, and coaching feedback.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/upload" className="cta-primary">
                Analyse a new run
              </Link>
              <Link href="/" className="cta-secondary">
                Back to coaching hub
              </Link>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="metric-tile">
              <p className="metric-value">{runs.length}</p>
              <p className="metric-label">Total runs in archive</p>
            </div>
            <div className="metric-tile">
              <p className="metric-value">{completedRuns.length}</p>
              <p className="metric-label">Completed recaps</p>
            </div>
            <div className="metric-tile">
              <p className="metric-value">{avgScore ?? '—'}</p>
              <p className="metric-label">Average score</p>
            </div>
            <div className="metric-tile">
              <p className="metric-value">{seasonGroups.length}</p>
              <p className="metric-label">{seasonGroups.length === 1 ? 'Season' : 'Seasons'} tracked</p>
            </div>
          </div>
        </div>
      </section>

      {!runs.length ? (
        <section className="surface-card p-6">
          <div className="surface-card-muted p-10 text-center">
            <div
              className="w-16 h-16 rounded-[1.4rem] flex items-center justify-center mx-auto mb-4"
              style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid var(--line-soft)' }}
            >
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--ink-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 8v4M12 16h.01"/>
              </svg>
            </div>
            <p className="text-base font-semibold" style={{ color: 'var(--ink-strong)' }}>No analyses yet</p>
            <p className="text-sm mt-2" style={{ color: 'var(--ink-soft)' }}>
              Upload a ski video to create your first recap card.
            </p>
          </div>
        </section>
      ) : (
        seasonGroups.map((group) => {
          const groupScored = group.runs.filter(
            (j): j is Job & { score: number } => (j as Job).status === 'done' && (j as Job).score != null,
          )
          const groupAvg = groupScored.length
            ? Math.round(groupScored.reduce((s, j) => s + j.score, 0) / groupScored.length)
            : null
          const groupBest = groupScored.length ? Math.max(...groupScored.map((j) => j.score)) : null

          return (
            <section key={group.label} className="surface-card p-6">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>
                      {group.runs.length} {group.runs.length === 1 ? 'run' : 'runs'}
                    </p>
                    <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                      {group.label}
                    </h2>
                  </div>
                  {groupAvg != null && (
                    <span className={levelBadgeClass(scoreLabel(groupAvg))} style={{ marginLeft: '0.5rem' }}>
                      {scoreLabel(groupAvg)}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  {groupAvg != null && (
                    <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                      Avg <span className="font-bold" style={{ color: 'var(--accent)' }}>{groupAvg}</span>
                    </span>
                  )}
                  {groupBest != null && (
                    <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                      Best <span className="font-bold" style={{ color: 'var(--success)' }}>{groupBest}</span>
                    </span>
                  )}
                </div>
              </div>

              <ul className="space-y-3 mt-5">
                {group.runs.map((run) => {
                  const job = run as Job
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
                        className="surface-card-muted flex items-center gap-4 px-5 py-4 group hover:-translate-y-0.5"
                        style={{ display: 'flex', transition: 'transform 150ms ease, background 0.15s ease, border-color 0.15s ease' }}
                      >
                        <div
                          className="w-11 h-11 rounded-2xl shrink-0 flex items-center justify-center"
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

                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-semibold truncate" style={{ color: 'var(--ink-strong)' }}>{filename}</p>
                          <p className="text-xs mt-1" style={{ color: 'var(--ink-soft)' }}>
                            {new Date(job.created_at).toLocaleString()}
                          </p>
                        </div>

                        {job.score != null && (
                          <span className="text-sm font-bold shrink-0" style={{ color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
                            {job.score}
                          </span>
                        )}

                        <span
                          className="shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full"
                          style={{ background: cfg.pill, color: cfg.dot }}
                        >
                          {cfg.label}
                        </span>

                        <svg
                          width="14" height="14"
                          viewBox="0 0 24 24" fill="none"
                          stroke="var(--ink-muted)"
                          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                          className="shrink-0 group-hover:translate-x-0.5"
                          style={{ transition: 'transform 150ms, stroke 150ms' }}
                        >
                          <path d="M9 18l6-6-6-6"/>
                        </svg>
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </section>
          )
        })
      )}
    </div>
  )
}
