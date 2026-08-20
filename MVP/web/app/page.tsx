import { createClient } from '@/lib/supabase/server'
import Link from 'next/link'
import { Job } from '@/lib/types'
import { buildTechniqueDashboard, isJudgeUnavailable, scoreCountsForProgress, scoreLabel, type AiCoachingPoint } from '@/lib/analysis-summary'
import { getDrill, localizeDrill } from '@/lib/drills'
import { SiteFooter } from '@/components/site-footer'
import { ScoreTrendCard } from '@/components/score-trend-card'
import { JobRetryAction } from '@/components/job-retry-action'
import { backfillMissingScores, loadAiCoachingForJobIds, loadPreviewUrlsForJobIds, loadSummariesForJobIds, resolveJobPresentation } from '@/lib/server-job-data'
import { getJobDisplayName, getJobUserNote, getJobOriginalFilename } from '@/lib/job-ui'
import { formatDate, getDictionary, translateKnownText, type Lang } from '@/lib/i18n'
import { readLanguage } from '@/lib/i18n-server'
import { SAMPLE_SUMMARY, SAMPLE_OVERLAY_PATH } from '@/lib/sample-run'

export const dynamic = 'force-dynamic'

const CATEGORY_BADGE: Record<string, string> = {
  balance: 'category-badge-balance',
  edging: 'category-badge-edging',
  rhythm: 'category-badge-rhythm',
  movement: 'category-badge-movement',
  general: 'category-badge-general',
}

const CATEGORY_ICON: Record<string, string> = {
  balance: 'Balance',
  edging: 'Edging',
  rhythm: 'Rhythm',
  movement: 'Movement',
  general: 'General',
}

const sampleDashboard = buildTechniqueDashboard(SAMPLE_SUMMARY)

/* ── Public Landing Page ──────────────────────────────── */
function LandingPage({ lang }: { lang: Lang }) {
  const dict = getDictionary(lang)

  return (
    <>
      <div className="route-bg route-bg--landing" />
      <div className="landing-scroll">

        {/* ═══ Section 1: Hero ═══════════════════════════ */}
        <section className="landing-section" style={{ minHeight: '85vh', display: 'flex', alignItems: 'center' }}>
          <div className="text-center mx-auto max-w-2xl">
            <span className="eyebrow" style={{ background: 'rgba(255,255,255,0.12)', borderColor: 'rgba(255,255,255,0.2)', color: '#fff' }}>
              {dict.landing.eyebrow}
            </span>
            <h1 className="landing-hero-title--white mt-6">
              {dict.landing.title}
            </h1>
            <p className="mt-5" style={{ fontSize: '1.1rem', lineHeight: 1.7, color: 'rgba(255,255,255,0.6)', maxWidth: '32rem', margin: '1.25rem auto 0' }}>
              {dict.landing.subtitle}
            </p>
            <div className="mt-8 flex flex-wrap gap-3 justify-center">
              <Link href="/signup" className="cta-primary" style={{ padding: '1rem 2rem', fontSize: '1rem' }}>
                {dict.landing.ctaPrimary}
              </Link>
              <Link
                href="/sample-analysis"
                className="cta-secondary"
                style={{ background: 'rgba(255,255,255,0.12)', color: '#fff', borderColor: 'rgba(255,255,255,0.2)', padding: '1rem 2rem' }}
              >
                {dict.landing.ctaSecondary}
              </Link>
            </div>
          </div>
        </section>

        {/* ═══ Section 2: Value props ════════════════════ */}
        <section className="landing-section">
          <div className="text-center mb-8">
            <p className="section-label">{dict.landing.whatYouGet}</p>
            <h2 className="section-title mt-3">{dict.landing.takeawaysTitle}</h2>
          </div>
          <div className="grid gap-5 md:grid-cols-3">
            <div className="surface-card p-7 text-center">
              <div className="step-number mx-auto" style={{ width: '3rem', height: '3rem', fontSize: '1rem' }}>1</div>
              <h3 className="mt-4 text-base font-bold" style={{ color: 'var(--ink-strong)' }}>{dict.landing.feature1Title}</h3>
              <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)', lineHeight: 1.6 }}>
                {dict.landing.feature1Body}
              </p>
            </div>
            <div className="surface-card p-7 text-center">
              <div className="step-number mx-auto" style={{ width: '3rem', height: '3rem', fontSize: '1rem' }}>2</div>
              <h3 className="mt-4 text-base font-bold" style={{ color: 'var(--ink-strong)' }}>{dict.landing.feature2Title}</h3>
              <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)', lineHeight: 1.6 }}>
                {dict.landing.feature2Body}
              </p>
            </div>
            <div className="surface-card p-7 text-center">
              <div className="step-number mx-auto" style={{ width: '3rem', height: '3rem', fontSize: '1rem' }}>3</div>
              <h3 className="mt-4 text-base font-bold" style={{ color: 'var(--ink-strong)' }}>{dict.landing.feature3Title}</h3>
              <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)', lineHeight: 1.6 }}>
                {dict.landing.feature3Body}
              </p>
            </div>
          </div>
        </section>

        {/* ═══ Section 3: How it works ═══════════════════ */}
        <section className="landing-section">
          <div className="surface-card p-8 lg:p-10">
            <div className="text-center mb-8">
              <p className="section-label">{dict.landing.howItWorks}</p>
              <h2 className="section-title mt-3">{dict.landing.howTitle}</h2>
            </div>
            <div className="grid gap-8 md:grid-cols-3">
              <div className="how-step">
                <span className="how-step-number">01</span>
                <h3>{dict.landing.how1Title}</h3>
                <p>{dict.landing.how1Body}</p>
              </div>
              <div className="how-step">
                <span className="how-step-number">02</span>
                <h3>{dict.landing.how2Title}</h3>
                <p>{dict.landing.how2Body}</p>
              </div>
              <div className="how-step">
                <span className="how-step-number">03</span>
                <h3>{dict.landing.how3Title}</h3>
                <p>{dict.landing.how3Body}</p>
              </div>
            </div>
          </div>
        </section>

        {/* ═══ Section 4: Sample preview ═════════════════ */}
        <section className="landing-section">
          <div className="text-center mb-6">
            <p className="section-label">{dict.landing.livePreview}</p>
            <h2 className="section-title mt-3">{dict.landing.previewTitle}</h2>
            <p className="section-copy mt-2 mx-auto max-w-lg">
              {dict.landing.previewBody}
            </p>
          </div>
          <div className="surface-card-strong p-5 lg:p-6 max-w-3xl mx-auto">
            <div
              className="overflow-hidden"
              style={{ borderRadius: 'var(--radius-xl)', background: '#0a0f1a', border: '1px solid rgba(255,255,255,0.15)' }}
            >
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video
                src={SAMPLE_OVERLAY_PATH}
                autoPlay
                loop
                muted
                playsInline
                className="w-full aspect-video bg-black"
              />
            </div>
            <div className="mt-4 grid gap-3 grid-cols-3">
              <div className="metric-tile metric-tile--high">
                <div className="metric-tile-dot" style={{ background: 'var(--accent)' }} />
                <p className="metric-value" style={{ color: 'var(--accent)', fontSize: '1.6rem' }}>{sampleDashboard.overview.overallScore}</p>
                <p className="metric-label">{dict.landing.techniqueScore}</p>
              </div>
              <div className="metric-tile">
                <p className="metric-value" style={{ fontSize: '1.6rem' }}>{sampleDashboard.overview.turnsDetected}</p>
                <p className="metric-label">{dict.landing.turnsDetected}</p>
              </div>
              <div className="metric-tile metric-tile--high">
                <div className="metric-tile-dot" style={{ background: 'var(--accent)' }} />
                <p className="metric-value" style={{ color: 'var(--accent)', fontSize: '1.6rem' }}>
                  {translateKnownText(sampleDashboard.overview.clipQualityLabel, lang)}
                </p>
                <p className="metric-label">{dict.landing.clipQuality}</p>
              </div>
            </div>
            <div className="mt-4 text-center">
              <Link href="/sample-analysis" className="cta-secondary" style={{ fontSize: '0.88rem' }}>
                {dict.landing.explore}
              </Link>
            </div>
          </div>
        </section>

        {/* ═══ Section 5: Final CTA ══════════════════════ */}
        <section className="landing-section">
          <div className="surface-card p-10 lg:p-14 text-center">
            <h2 className="section-title">{dict.landing.finalTitle}</h2>
            <p className="section-copy mt-3 mx-auto max-w-lg">
              {dict.landing.finalBody}
            </p>
            <div className="mt-8 flex flex-wrap gap-3 justify-center">
              <Link href="/signup" className="cta-primary" style={{ padding: '1rem 2rem', fontSize: '1rem' }}>
                {dict.landing.ctaPrimary}
              </Link>
              <Link href="/sample-analysis" className="cta-secondary" style={{ padding: '1rem 2rem' }}>
                {dict.landing.viewSample}
              </Link>
            </div>
          </div>
        </section>

        <SiteFooter lang={lang} />
      </div>
    </>
  )
}

/* ── Authenticated Dashboard ──────────────────────────── */
async function Dashboard({ lang }: { lang: Lang }) {
  const dict = getDictionary(lang)
  const supabase = createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  const { createServiceClient } = await import('@/lib/supabase/server')
  const service = createServiceClient()

  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .order('created_at', { ascending: false })

  const runs = (jobs ?? []) as Job[]
  const completedRuns = runs.filter((j) => j.status === 'done')
  await backfillMissingScores(service, completedRuns)
  const latestCompleted = completedRuns[0] ?? null

  const scoredRuns = completedRuns.filter((j) => j.score != null) as (Job & { score: number })[]
  const summaryByJob = await loadSummariesForJobIds(service, completedRuns.map((run) => run.id))
  const progressScoredRuns = scoredRuns.filter((run) => scoreCountsForProgress(summaryByJob.get(run.id)))
  const latestScore = progressScoredRuns[0]?.score ?? null
  const bestRecentScore = progressScoredRuns.length ? Math.max(...progressScoredRuns.slice(0, 10).map((j) => j.score)) : null

  let score = latestScore
  let primaryJudgePoint: AiCoachingPoint | null = null
  const recentCompleted = completedRuns.slice(0, 3)
  const judgeByJob = await loadAiCoachingForJobIds(service, recentCompleted.map((run) => run.id))

  for (const run of recentCompleted) {
    const judge = judgeByJob.get(run.id)
    if (!judge || isJudgeUnavailable(judge)) continue
    primaryJudgePoint = judge.coaching_points?.[0] ?? null
    if (primaryJudgePoint) break
  }

  if (score == null && latestCompleted) {
    const latestSummary = summaryByJob.get(latestCompleted.id)
    if (latestSummary && scoreCountsForProgress(latestSummary)) {
      const dashboard = buildTechniqueDashboard(latestSummary)
      score = dashboard.overview.overallScore
      if (Number.isFinite(score)) {
        await service.from('jobs').update({ score }).eq('id', latestCompleted.id)
      }
    }
  }

  const judgeDrills = primaryJudgePoint?.recommended_drill_id
    ? [getDrill(primaryJudgePoint.recommended_drill_id)].filter((drill): drill is NonNullable<typeof drill> => drill != null)
    : []
  const recentRuns = runs.slice(0, 5)
  const recentPreviewUrlByJob = await loadPreviewUrlsForJobIds(service, recentRuns.map((run) => run.id))
  const displayName = user?.email?.split('@')[0] ?? (lang === 'zh' ? '你' : 'there')
  const trendRuns = progressScoredRuns.slice(0, 10)

  return (
    <>
      <div className="space-y-6">

        {/* ══ ROW 1: Welcome | Upload | Preflight ══════════ */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Welcome */}
          <section className="surface-card p-8">
            <h1 style={{ fontSize: 'clamp(1.5rem, 2.4vw, 2.2rem)', fontWeight: 800, lineHeight: 1.15, letterSpacing: '-0.03em', color: 'var(--ink-strong)' }}>
              {dict.dashboard.welcomeBack}, <span style={{ color: 'var(--accent)' }}>{displayName}</span>.
            </h1>
            <p className="section-copy mt-2">
              {completedRuns.length > 0
                ? dict.dashboard.ready
                : dict.dashboard.firstSession}
            </p>
          </section>

          {/* Upload */}
          <Link href="/upload" className="upload-card">
            <div className="upload-card-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
              </svg>
            </div>
            <h3>{completedRuns.length > 0 ? dict.dashboard.uploadNext : dict.dashboard.uploadFirst}</h3>
            <p>{dict.dashboard.supportedFormats}</p>
          </Link>

          {/* Preflight checklist */}
          <section className="surface-card-strong p-6">
            <p className="section-label">{dict.dashboard.preflight}</p>
            <div className="mt-4 space-y-3">
              <div className="preflight-item">
                <span className="preflight-number">01</span>
                <div>
                  <h4>{dict.dashboard.preflight1Title}</h4>
                  <p>{dict.dashboard.preflight1Body}</p>
                </div>
              </div>
              <div className="preflight-item">
                <span className="preflight-number">02</span>
                <div>
                  <h4>{dict.dashboard.preflight2Title}</h4>
                  <p>{dict.dashboard.preflight2Body}</p>
                </div>
              </div>
              <div className="preflight-item">
                <span className="preflight-number">03</span>
                <div>
                  <h4>{dict.dashboard.preflight3Title}</h4>
                  <p>{dict.dashboard.preflight3Body}</p>
                </div>
              </div>
            </div>
          </section>
        </div>

        <ScoreTrendCard
          runs={trendRuns}
          title={dict.dashboard.trendTitle}
          subtitle={dict.dashboard.trendSubtitle}
          lang={lang}
        />

        {/* ══ ROW 2: Metrics+Runs | Coaching Insight | Practice ══ */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left: Metric tiles + Recent runs */}
          <div className="space-y-6">
            {/* Metric tiles */}
            <div className="grid gap-3 grid-cols-3">
              <div className="metric-tile">
                <p className="metric-value">{score ?? '—'}</p>
                <p className="metric-label">{dict.dashboard.techniqueScore}</p>
              </div>
              <div className="metric-tile">
                <p className="metric-value">{progressScoredRuns.length > 0 ? progressScoredRuns.length : '—'}</p>
                <p className="metric-label">{dict.dashboard.scoredRuns}</p>
              </div>
              <div className="metric-tile">
                <p className="metric-value">{bestRecentScore ?? '—'}</p>
                <p className="metric-label">{dict.dashboard.bestRecent}</p>
              </div>
            </div>

            {/* Recent runs */}
            <section className="surface-card p-6">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <p className="section-label">{dict.dashboard.recentRuns}</p>
                <Link href="/jobs" className="cta-secondary text-sm" style={{ padding: '0.4rem 0.8rem', fontSize: '0.78rem' }}>
                  {dict.dashboard.viewAll}
                </Link>
              </div>

              {!recentRuns.length ? (
                <div className="surface-card-muted p-6 text-center mt-4">
                  <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                    {dict.dashboard.noRuns}
                  </p>
                </div>
              ) : (
                <ul className="space-y-2 mt-4">
                  {recentRuns.map((job: Job) => {
                    const displayName = getJobDisplayName(job)
                    const originalFilename = getJobOriginalFilename(job)
                    const userNote = getJobUserNote(job)
                    const previewUrl = recentPreviewUrlByJob.get(job.id) ?? null
                    const presentation = resolveJobPresentation(job)
                    const excludedFromProgress = job.score != null && !scoreCountsForProgress(summaryByJob.get(job.id))
                    return (
                      <li key={job.id}>
                        <div
                          className="surface-card-muted flex items-center gap-3 px-4 py-3 group transition-transform hover:-translate-y-0.5"
                          style={{ display: 'flex' }}
                        >
                          <Link href={`/jobs/${job.id}`} className="flex min-w-0 flex-1 items-center gap-3">
                            {previewUrl ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img
                                src={previewUrl}
                                alt={displayName}
                                className="rounded-[var(--radius-lg)] shrink-0"
                                style={{ width: '4.25rem', height: '4.25rem', objectFit: 'cover' }}
                              />
                            ) : (
                              <div
                                className="rounded-[var(--radius-lg)] flex items-center justify-center shrink-0"
                                style={{ width: '4.25rem', height: '4.25rem', background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.06)' }}
                              >
                                <div className="w-2.5 h-2.5 rounded-full" style={{ background: presentation.dot }} />
                              </div>
                            )}
                            <div className="min-w-0 flex-1">
                              <p className="text-sm font-semibold truncate" style={{ color: 'var(--ink-strong)' }}>
                                {displayName}
                              </p>
                              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs" style={{ color: 'var(--ink-muted)' }}>
                                {originalFilename && originalFilename !== displayName && <span className="truncate">{originalFilename}</span>}
                                <span>{formatDate(job.created_at, lang)}</span>
                              </div>
                              {userNote && (
                                <p className="mt-1 text-xs truncate" style={{ color: 'var(--ink-soft)' }}>
                                  {userNote}
                                </p>
                              )}
                            </div>
                          </Link>

                          <div className="flex items-center gap-2 shrink-0">
                            {job.score != null && (
                              <span className="text-xs font-bold shrink-0" style={{ color: 'var(--accent)' }}>
                                {job.score}
                              </span>
                            )}
                            {excludedFromProgress && (
                              <span className="text-[0.68rem] font-semibold px-2 py-0.5 rounded-full shrink-0" style={{ color: 'var(--gold)', background: 'var(--gold-dim)' }}>
                                {dict.dashboard.excludedFromProgress}
                              </span>
                            )}
                            <span
                              className="text-xs font-semibold px-2.5 py-1 rounded-full shrink-0"
                              style={{ color: presentation.dot, background: presentation.pill }}
                            >
                              {translateKnownText(presentation.label, lang)}
                            </span>
                            <JobRetryAction
                              jobId={job.id}
                              canRetry={presentation.retryable}
                              actionLabel={presentation.actionLabel}
                              compact
                            />
                          </div>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
            </section>
          </div>

          {/* Middle: Latest coaching insight */}
          <section className="surface-card p-6">
            <p className="section-label">{dict.dashboard.latestInsight}</p>
            <h2 className="mt-3" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
              {primaryJudgePoint
                ? primaryJudgePoint.title
                : completedRuns.length > 0
                  ? dict.dashboard.latestReady
                  : dict.dashboard.uploadToStart}
            </h2>
            {primaryJudgePoint && (
              <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-soft)' }}>
                {primaryJudgePoint.feedback}
              </p>
            )}
            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/upload" className="cta-primary" style={{ padding: '0.75rem 1.2rem', fontSize: '0.88rem' }}>
                {dict.dashboard.analyzeNew}
              </Link>
              {latestCompleted && (
                <Link href={`/jobs/${latestCompleted.id}`} className="cta-secondary" style={{ padding: '0.75rem 1.2rem', fontSize: '0.88rem' }}>
                  {dict.dashboard.openRecap}
                </Link>
              )}
            </div>
          </section>

          {/* Right: Practice focus */}
          <div className="space-y-6">
            {(judgeDrills.length > 0 || primaryJudgePoint) && (
              <section className="surface-card p-6">
                <p className="section-label">{dict.dashboard.practiceFocus}</p>
                <div className="mt-4 space-y-3">
                  {judgeDrills.slice(0, 3).map((drill) => {
                    const localizedDrill = localizeDrill(drill, lang)
                    return (
                      <div key={drill.id} className={`coaching-card coaching-accent-${drill.category}`}>
                        <p className="text-sm font-bold pl-3" style={{ color: 'var(--ink-strong)' }}>
                          {localizedDrill.title}
                        </p>
                        <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-soft)' }}>
                          {localizedDrill.description}
                        </p>
                        <div className="mt-3 pl-3">
                          <span
                            className={`text-xs font-semibold px-2 py-0.5 rounded-full ${CATEGORY_BADGE[drill.category] ?? 'category-badge-general'}`}
                          >
                            {translateKnownText(CATEGORY_ICON[drill.category] ?? drill.category, lang)}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                  {primaryJudgePoint && judgeDrills.length === 0 && (
                    <div className="coaching-card">
                      <p className="text-sm font-bold pl-3" style={{ color: 'var(--ink-strong)' }}>
                        {primaryJudgePoint.title}
                      </p>
                      <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-soft)' }}>
                        {primaryJudgePoint.feedback}
                      </p>
                    </div>
                  )}
                </div>
              </section>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

/* ── Main page component ──────────────────────────────── */
export default async function HomePage() {
  const lang = readLanguage()
  const supabase = createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    return <LandingPage lang={lang} />
  }

  return <Dashboard lang={lang} />
}
