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

  // Use session client for user-facing job reads — RLS scopes by user_id
  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .order('created_at', { ascending: false })

  const runs = (jobs ?? []) as Job[]
  const completedRuns = runs.filter((j) => j.status === 'done')
  const latestCompleted = completedRuns[0] ?? null

  // Build scored runs list from stored scores
  const scoredRuns = completedRuns.filter((j) => j.score != null) as (Job & { score: number })[]
  const latestScore = scoredRuns[0]?.score ?? null
  const previousScore = scoredRuns[1]?.score ?? null
  const scoreDelta = latestScore != null && previousScore != null ? latestScore - previousScore : null
  const bestRecentScore = scoredRuns.length ? Math.max(...scoredRuns.slice(0, 10).map((j) => j.score)) : null
  const recentTrend = scoredRuns.slice(0, 5).map((j) => j.score)

  // Fallback: if no stored score, try computing from summary
  let score = latestScore
  let level: string | null = latestScore != null ? scoreLabel(latestScore) : null
  let primaryTip: CoachingTip | null = null

  // Fetch summaries from last 3 completed runs in parallel for coaching patterns
  const recentCompleted = completedRuns.slice(0, 3)
  const recentTipSets: CoachingTip[][] = []

  const summaries = await Promise.all(
    recentCompleted.map((run) => fetchSummary(service, run))
  )

  for (let i = 0; i < recentCompleted.length; i++) {
    const run = recentCompleted[i]
    const summary = summaries[i]
    if (!summary) continue

    // Collect coaching tips for practice guidance
    const tips = summary.coaching_tips ?? []
    if (tips.length) recentTipSets.push(tips)

    // Handle latest run specifically
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
        // Persist score for future
        if (Number.isFinite(score)) {
          await service.from('jobs').update({ score }).eq('id', run.id)
        }
      } else {
        // Already have score, just need the tip
        const dashboard = buildTechniqueDashboard(summary)
        if (dashboard.focusCards.length) {
          primaryTip = dashboard.focusCards[0]
        } else if (tips.length) {
          primaryTip = tips[0]
        }
      }
    }
  }

  // Build next-session coaching card from recurring patterns
  const nextSession = buildNextSessionCard(recentTipSets)

  const recentRuns = runs.slice(0, 5)

  return (
    <div className="space-y-6">
      {/* ── Hero section ─────────────────────────────── */}
      <section className="surface-card hero-arc-bg p-8 lg:p-10">
        <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <span className="eyebrow">Your coaching hub</span>
            <h1 className="section-title mt-6">
              {score != null
                ? 'Keep the momentum. Your technique is progressing.'
                : latestCompleted
                  ? 'Your latest run is ready to review.'
                  : 'Upload your first run to start coaching.'}
            </h1>
            <p className="section-copy mt-4 max-w-xl">
              {score != null
                ? 'Review your latest analysis, track your level, and focus on what matters most for your next session.'
                : 'Drop a video from the slopes and get back an AI-powered recap with technique scores, key moments, and practice priorities.'}
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/upload" className="cta-primary">
                Analyse a new run
              </Link>
              <Link href="/jobs" className="cta-secondary">
                {runs.length ? `View archive` : 'Start your archive'}
              </Link>
            </div>
          </div>

          {/* Score ring or empty state */}
          <div className="flex items-center justify-center">
            {score != null && level != null ? (
              <div className="text-center">
                <div className="score-ring mx-auto" style={{ width: '11rem', height: '11rem' }}>
                  <div className="score-ring-glow" />
                  <svg width="176" height="176" viewBox="0 0 176 176">
                    <circle cx="88" cy="88" r="74" fill="none" stroke="rgba(31,42,51,0.06)" strokeWidth="8" />
                    <circle
                      cx="88" cy="88" r="74"
                      fill="none"
                      stroke="url(#scoreGradHome)"
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray="464.96"
                      strokeDashoffset={464.96 - (score / 100) * 464.96}
                    />
                    <defs>
                      <linearGradient id="scoreGradHome" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="#4f8fb3" />
                        <stop offset="100%" stopColor="#c79a44" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="score-ring-label">
                    <span
                      className="font-bold tracking-tight"
                      style={{ fontSize: 'clamp(2rem, 5vw, 3.5rem)', fontWeight: 700, color: 'var(--ink-strong)' }}
                    >
                      {score}
                    </span>
                    <span className="text-xs mt-1" style={{ color: 'var(--ink-soft)' }}>
                      technique
                    </span>
                  </div>
                </div>
                <div className="mt-4 flex items-center justify-center gap-2">
                  <span className={levelBadgeClass(level)}>{level}</span>
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
                <p className="text-sm mt-3" style={{ color: 'var(--ink-soft)' }}>
                  Latest completed run
                </p>
              </div>
            ) : (
              <div
                className="surface-card-muted p-8 text-center w-full"
                style={{ maxWidth: '20rem' }}
              >
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                  style={{ background: 'rgba(31,42,51,0.03)', border: '1px solid var(--line-soft)' }}
                >
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--ink-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                  </svg>
                </div>
                <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
                  {latestCompleted ? 'Summary pending' : 'No completed runs yet'}
                </p>
                <p className="text-sm mt-2" style={{ color: 'var(--ink-soft)' }}>
                  {latestCompleted
                    ? 'Your run completed but the summary data is still processing.'
                    : 'Your technique score and level will appear here after your first analysis.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ── Progression widgets ──────────────────────── */}
      {scoredRuns.length > 0 && (
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
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
            {recentTrend.length > 1 ? (
              <div className="flex items-end gap-1 h-8">
                {[...recentTrend].reverse().map((s, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-sm"
                    style={{
                      height: `${Math.max(20, (s / 100) * 100)}%`,
                      background: i === recentTrend.length - 1 ? 'var(--accent)' : 'rgba(31,42,51,0.08)',
                    }}
                  />
                ))}
              </div>
            ) : (
              <p className="metric-value">—</p>
            )}
            <p className="metric-label">Recent trend ({recentTrend.length} runs)</p>
          </div>
        </section>
      )}

      {/* ── Coaching focus + next session ─────────────── */}
      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        {/* Primary coaching focus */}
        <section className="surface-card-strong p-6">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Primary focus</p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                {primaryTip ? primaryTip.title : 'Your next focus'}
              </h2>
            </div>
            <span className="status-pill" style={{ color: 'var(--accent)', background: 'var(--accent-dim)' }}>
              Coaching
            </span>
          </div>

          {primaryTip ? (
            <div className="mt-5">
              <div className="coaching-card">
                <p className="text-base leading-7 pl-3" style={{ color: 'var(--ink-base)' }}>
                  {primaryTip.explanation}
                </p>
                <p className="mt-3 text-xs pl-3" style={{ color: 'var(--ink-muted)' }}>
                  {primaryTip.evidence}
                </p>
              </div>

              {latestCompleted && (
                <Link href={`/jobs/${latestCompleted.id}`} className="cta-secondary w-full mt-4">
                  Open full run recap
                </Link>
              )}
            </div>
          ) : (
            <div className="mt-5 surface-card-muted p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
              {latestCompleted
                ? 'No coaching tips are available for this run. Open the recap for full details.'
                : 'Upload and complete your first run to receive personalised coaching focus cards.'}
              {latestCompleted && (
                <Link href={`/jobs/${latestCompleted.id}`} className="cta-secondary w-full mt-4">
                  Open run recap
                </Link>
              )}
            </div>
          )}
        </section>

        {/* Next session coaching card */}
        <section className="surface-card p-6">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>What next?</p>
              <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                Next session
              </h2>
            </div>
            <span className="status-pill" style={{ color: 'var(--gold)', background: 'var(--gold-dim)' }}>
              Practice plan
            </span>
          </div>

          <p className="mt-4 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
            {nextSession.headline}
          </p>

          {nextSession.drills.length > 0 ? (
            <div className="mt-5 space-y-3">
              {nextSession.drills.map((drill) => (
                <div key={drill.id} className={`coaching-card coaching-accent-${drill.category}`}>
                  <div className="flex items-center justify-between gap-3 pl-3">
                    <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
                      {drill.title}
                    </p>
                    <span
                      className={`text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 ${CATEGORY_BADGE[drill.category] ?? 'category-badge-general'}`}
                    >
                      {CATEGORY_ICON[drill.category] ?? drill.category}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-soft)' }}>
                    {drill.description}
                  </p>
                  {drill.priority > 1 && (
                    <p className="mt-2 text-xs font-semibold pl-3" style={{ color: 'var(--gold)' }}>
                      Recurring across {drill.priority} recent runs
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-5 surface-card-muted p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
              Upload more runs to build your personalised practice plan. We look for patterns across your recent sessions to suggest drills.
            </div>
          )}
        </section>
      </div>

      {/* ── Archive preview ──────────────────────────── */}
      <section className="surface-card p-6">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Recent runs</p>
            <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
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
  )
}
