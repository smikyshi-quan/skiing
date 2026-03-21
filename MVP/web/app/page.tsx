import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'
import { Job, JobStatus } from '@/lib/types'
import { buildTechniqueDashboard, scoreLabel, type TechniqueRunSummary, type CoachingTip } from '@/lib/analysis-summary'
import { buildNextSessionCard } from '@/lib/practice-guidance'

export const dynamic = 'force-dynamic'

const STATUS_DOT: Record<JobStatus, string> = {
  created: 'var(--ink-muted)',
  uploaded: 'var(--accent)',
  queued: 'var(--gold)',
  running: 'var(--accent)',
  done: 'var(--success)',
  error: 'var(--danger)',
}

const STATUS_LABEL: Record<JobStatus, string> = {
  created: 'Created',
  uploaded: 'Uploaded',
  queued: 'Queued',
  running: 'Analysing',
  done: 'Done',
  error: 'Error',
}

const CATEGORY_ICON: Record<string, string> = {
  balance: 'Balance',
  edging: 'Edging',
  rhythm: 'Rhythm',
  movement: 'Movement',
  general: 'General',
}

const CATEGORY_BADGE: Record<string, string> = {
  balance: 'category-badge-balance',
  edging: 'category-badge-edging',
  rhythm: 'category-badge-rhythm',
  movement: 'category-badge-movement',
  general: 'category-badge-general',
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

async function fetchSummary(
  service: ReturnType<typeof import('@/lib/supabase/server').createServiceClient>,
  job: Job,
): Promise<TechniqueRunSummary | null> {
  const { data: artifacts } = await service
    .from('artifacts')
    .select('*')
    .eq('job_id', job.id)
    .eq('kind', 'summary_json')
    .limit(1)

  const summaryArtifact = artifacts?.[0]
  if (!summaryArtifact) return null

  const { data: file } = await service.storage
    .from('artifacts')
    .download(summaryArtifact.object_path)

  if (!file) return null

  try {
    return JSON.parse(await file.text()) as TechniqueRunSummary
  } catch {
    return null
  }
}

export default async function HomePage() {
  const supabase = createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) redirect('/login')

  const { createServiceClient } = await import('@/lib/supabase/server')
  const service = createServiceClient()

  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .order('created_at', { ascending: false })

  const runs = (jobs ?? []) as Job[]
  const completedRuns = runs.filter((j) => j.status === 'done')
  const latestCompleted = completedRuns[0] ?? null

  const scoredRuns = completedRuns.filter((j) => j.score != null) as (Job & { score: number })[]
  const latestScore = scoredRuns[0]?.score ?? null
  const previousScore = scoredRuns[1]?.score ?? null
  const scoreDelta = latestScore != null && previousScore != null ? latestScore - previousScore : null
  const bestRecentScore = scoredRuns.length ? Math.max(...scoredRuns.slice(0, 10).map((j) => j.score)) : null

  let score = latestScore
  let level: string | null = latestScore != null ? scoreLabel(latestScore) : null
  let primaryTip: CoachingTip | null = null

  const recentCompleted = completedRuns.slice(0, 3)
  const recentTipSets: CoachingTip[][] = []

  const summaries = await Promise.all(
    recentCompleted.map((run) => fetchSummary(service, run))
  )

  for (let i = 0; i < recentCompleted.length; i++) {
    const run = recentCompleted[i]
    const summary = summaries[i]
    if (!summary) continue

    const tips = summary.coaching_tips ?? []
    if (tips.length) recentTipSets.push(tips)

    if (run.id === latestCompleted?.id) {
      if (score == null) {
        const dashboard = buildTechniqueDashboard(summary)
        score = dashboard.overview.overallScore
        level = scoreLabel(score)
        if (dashboard.focusCards.length) {
          primaryTip = dashboard.focusCards[0]
        } else if (tips.length) {
          primaryTip = tips[0]
        }
        if (Number.isFinite(score)) {
          await service.from('jobs').update({ score }).eq('id', run.id)
        }
      } else {
        const dashboard = buildTechniqueDashboard(summary)
        if (dashboard.focusCards.length) {
          primaryTip = dashboard.focusCards[0]
        } else if (tips.length) {
          primaryTip = tips[0]
        }
      }
    }
  }

  const nextSession = buildNextSessionCard(recentTipSets)
  const recentRuns = runs.slice(0, 5)

  return (
    <>
      <div className="route-bg route-bg--dashboard" />
      <div className="space-y-6">
        {/* ── Hero: Insight panel + Practice cards ────── */}
        <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          {/* What stands out first */}
          <section className="surface-card p-8 lg:p-10">
            <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>What stands out first</p>

            <h1
              className="mt-5"
              style={{ fontSize: 'clamp(1.6rem, 2.8vw, 2.4rem)', fontWeight: 800, lineHeight: 1.15, letterSpacing: '-0.03em', color: 'var(--ink-strong)' }}
            >
              {primaryTip
                ? primaryTip.explanation
                : score != null
                  ? 'Your latest analysis is ready. Review your technique below.'
                  : 'Upload your first run to start coaching.'}
            </h1>

            {/* Stat tiles */}
            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              <div className="metric-tile">
                <div className="flex items-center justify-between">
                  <p className="metric-value">{score ?? '—'}</p>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 15l-3-3m0 0l3-3m-3 3h8M3 12a9 9 0 1118 0 9 9 0 01-18 0z" />
                  </svg>
                </div>
                <p className="metric-label">Best single-turn quality score</p>
              </div>
              <div className="metric-tile">
                <div className="flex items-center justify-between">
                  <p className="metric-value">{scoredRuns.length > 0 ? scoredRuns.length : '—'}</p>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <p className="metric-label">Runs with technique scores</p>
              </div>
              <div className="metric-tile">
                <div className="flex items-center justify-between">
                  <p className="metric-value">{completedRuns.length}</p>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                </div>
                <p className="metric-label">Artifacts ready to inspect or export</p>
              </div>
              <div className="metric-tile">
                <div className="flex items-center justify-between">
                  <p className="metric-value">{bestRecentScore ?? '—'}</p>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" />
                  </svg>
                </div>
                <p className="metric-label">Key images surfaced from the run</p>
              </div>
            </div>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/upload" className="cta-primary">
                Analyse a new run
              </Link>
              {latestCompleted && (
                <Link href={`/jobs/${latestCompleted.id}`} className="cta-secondary">
                  Open full run recap
                </Link>
              )}
            </div>
          </section>

          {/* Practice cards */}
          <section className="surface-card-strong p-6 lg:p-8 self-start">
            <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Practice Cards</p>

            <div className="mt-5 space-y-4">
              {(nextSession.drills.length > 0 ? nextSession.drills : []).slice(0, 3).map((drill) => (
                <div key={drill.id} className={`coaching-card coaching-accent-${drill.category}`}>
                  <p className="text-sm font-bold pl-3" style={{ color: 'var(--ink-strong)' }}>
                    {drill.title}
                  </p>
                  <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-soft)' }}>
                    {drill.description}
                  </p>
                  <div className="mt-3 pl-3 flex items-center justify-between">
                    <span
                      className={`text-xs font-semibold px-2 py-0.5 rounded-full ${CATEGORY_BADGE[drill.category] ?? 'category-badge-general'}`}
                    >
                      {CATEGORY_ICON[drill.category] ?? drill.category}
                    </span>
                    {latestCompleted && (
                      <Link
                        href={`/jobs/${latestCompleted.id}`}
                        className="inline-flex items-center gap-1 text-xs font-semibold"
                        style={{ color: 'var(--ink-soft)' }}
                      >
                        Watch
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                      </Link>
                    )}
                  </div>
                </div>
              ))}

              {primaryTip && (
                <div className="coaching-card">
                  <p className="text-sm font-bold pl-3" style={{ color: 'var(--ink-strong)' }}>
                    {primaryTip.title}
                  </p>
                  <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-soft)' }}>
                    {primaryTip.explanation}
                  </p>
                  {latestCompleted && (
                    <div className="mt-3 pl-3">
                      <Link
                        href={`/jobs/${latestCompleted.id}`}
                        className="inline-flex items-center gap-1 text-xs font-semibold"
                        style={{ color: 'var(--ink-soft)' }}
                      >
                        Watch
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                      </Link>
                    </div>
                  )}
                </div>
              )}

              {!primaryTip && nextSession.drills.length === 0 && (
                <div className="surface-card-muted p-6 text-sm text-center" style={{ color: 'var(--ink-soft)' }}>
                  Upload and complete your first run to receive personalised coaching focus cards.
                </div>
              )}
            </div>
          </section>
        </div>

        {/* ── Progression + Score ring ────────────────── */}
        {scoredRuns.length > 0 && (
          <div className="grid gap-6 lg:grid-cols-[0.4fr_1fr]">
            <section className="surface-card p-6 flex flex-col items-center justify-center">
              <div className="score-ring mx-auto" style={{ width: '10rem', height: '10rem' }}>
                <div className="score-ring-glow" />
                <svg width="160" height="160" viewBox="0 0 160 160">
                  <circle cx="80" cy="80" r="68" fill="none" stroke="rgba(0,0,0,0.06)" strokeWidth="7" />
                  <circle
                    cx="80" cy="80" r="68"
                    fill="none"
                    stroke="url(#scoreGradHome)"
                    strokeWidth="7"
                    strokeLinecap="round"
                    strokeDasharray="427.26"
                    strokeDashoffset={427.26 - ((score ?? 0) / 100) * 427.26}
                  />
                  <defs>
                    <linearGradient id="scoreGradHome" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#0084d4" />
                      <stop offset="100%" stopColor="#c79a44" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="score-ring-label">
                  <span className="font-extrabold tracking-tight" style={{ fontSize: '2.5rem', color: 'var(--ink-strong)' }}>
                    {score}
                  </span>
                  <span className="text-xs mt-1" style={{ color: 'var(--ink-soft)' }}>technique</span>
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2">
                {level && <span className={levelBadgeClass(level)}>{level}</span>}
                {scoreDelta != null && (
                  <span
                    className="text-xs font-bold px-2 py-0.5 rounded-full"
                    style={{
                      color: scoreDelta >= 0 ? 'var(--success)' : 'var(--danger)',
                      background: scoreDelta >= 0 ? 'var(--success-dim)' : 'var(--danger-dim)',
                    }}
                  >
                    {scoreDelta >= 0 ? '+' : ''}{scoreDelta}
                  </span>
                )}
              </div>
            </section>

            <section className="surface-card p-6">
              <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Progression</p>
              <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div className="metric-tile">
                  <p className="metric-value">{latestScore}</p>
                  <p className="metric-label">Latest score</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value" style={{ color: scoreDelta != null && scoreDelta >= 0 ? 'var(--success)' : scoreDelta != null ? 'var(--danger)' : 'var(--ink-strong)' }}>
                    {scoreDelta != null ? `${scoreDelta >= 0 ? '+' : ''}${scoreDelta}` : '—'}
                  </p>
                  <p className="metric-label">Delta vs previous</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{bestRecentScore ?? '—'}</p>
                  <p className="metric-label">Best recent run</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{completedRuns.length}</p>
                  <p className="metric-label">Completed recaps</p>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ── Archive preview ──────────────────────────── */}
        <section className="surface-card p-6">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/alpine/hero-powder.jpg"
            alt="Fresh powder tracks on the mountain"
            className="hero-photo mb-5"
            style={{ height: '140px' }}
          />
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Recent runs</p>
              <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                Archive
              </h2>
            </div>
            <Link href="/jobs" className="cta-secondary text-sm" style={{ padding: '0.5rem 0.9rem' }}>
              View all
            </Link>
          </div>

          {!recentRuns.length ? (
            <div className="surface-card-muted p-8 text-center mt-5">
              <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                Your runs will appear here.
              </p>
            </div>
          ) : (
            <ul className="space-y-2 mt-5">
              {recentRuns.map((job: Job) => {
                const filename =
                  String(job.config?.original_filename ?? '') ||
                  job.video_object_path?.split('/').pop() ||
                  job.id.slice(0, 8)
                return (
                  <li key={job.id}>
                    <Link
                      href={`/jobs/${job.id}`}
                      className="surface-card-muted flex items-center gap-3 px-4 py-3 group transition-transform hover:-translate-y-0.5"
                      style={{ display: 'flex' }}
                    >
                      <div
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ background: STATUS_DOT[job.status as JobStatus] }}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate" style={{ color: 'var(--ink-strong)' }}>
                          {filename}
                        </p>
                      </div>
                      {job.score != null && (
                        <span className="text-xs font-bold shrink-0" style={{ color: 'var(--accent)' }}>
                          {job.score}
                        </span>
                      )}
                      <span className="text-xs shrink-0" style={{ color: 'var(--ink-muted)' }}>
                        {STATUS_LABEL[job.status as JobStatus]}
                      </span>
                      <span className="text-xs shrink-0" style={{ color: 'var(--ink-muted)' }}>
                        {new Date(job.created_at).toLocaleDateString()}
                      </span>
                    </Link>
                  </li>
                )
              })}
            </ul>
          )}
        </section>
      </div>
    </>
  )
}
