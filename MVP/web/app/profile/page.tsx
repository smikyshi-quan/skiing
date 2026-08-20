import { createClient, createServiceClient } from '@/lib/supabase/server'
import Link from 'next/link'
import { redirect } from 'next/navigation'
import type { Job } from '@/lib/types'
import { scoreCountsForProgress, scoreLabel } from '@/lib/analysis-summary'
import { backfillMissingScores, loadPreviewUrlsForJobIds, loadSummariesForJobIds, resolveJobPresentation } from '@/lib/server-job-data'
import { getJobDisplayName, getJobUserNote, getJobOriginalFilename } from '@/lib/job-ui'
import { ScoreTrendCard } from '@/components/score-trend-card'
import { RunMetadataEditor } from '@/components/run-metadata-editor'
import { JobRetryAction } from '@/components/job-retry-action'
import { formatDate, formatDateTime, getDictionary, translateKnownText } from '@/lib/i18n'
import { readLanguage } from '@/lib/i18n-server'

export const dynamic = 'force-dynamic'

export default async function ProfilePage() {
  const lang = readLanguage()
  const dict = getDictionary(lang)
  const supabase = createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) redirect('/login')

  const service = createServiceClient()
  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .order('created_at', { ascending: false })

  const runs = (jobs ?? []) as Job[]
  const completedRuns = runs.filter((job) => job.status === 'done')
  await backfillMissingScores(service, completedRuns)
  const scoredRuns = completedRuns.filter((job): job is Job & { score: number } => job.score != null)
  const summaryByJob = await loadSummariesForJobIds(service, completedRuns.map((job) => job.id))
  const progressScoredRuns = scoredRuns.filter((job) => scoreCountsForProgress(summaryByJob.get(job.id)))
  const avgScore = progressScoredRuns.length
    ? Math.round(progressScoredRuns.reduce((sum, job) => sum + job.score, 0) / progressScoredRuns.length)
    : null
  const bestScore = progressScoredRuns.length ? Math.max(...progressScoredRuns.map((job) => job.score)) : null
  const latestCompleted = completedRuns[0] ?? null
  const latestCompletedExcluded = latestCompleted?.score != null
    && !scoreCountsForProgress(summaryByJob.get(latestCompleted.id))
  const bestRun = progressScoredRuns.length
    ? progressScoredRuns.reduce((best, run) => (best == null || run.score > best.score ? run : best), null as (Job & { score: number }) | null)
    : null

  const displayName = user.email?.split('@')[0] ?? 'Athlete'
  const initials = displayName.slice(0, 2).toUpperCase()
  const memberSince = formatDate(user.created_at, lang, {
    month: 'long',
    year: 'numeric',
  })

  const recentRuns = runs.slice(0, 8)
  const previewUrlByJob = await loadPreviewUrlsForJobIds(service, recentRuns.map((job) => job.id))

  return (
    <div className="space-y-6">
      <section className="surface-card p-8 lg:p-10">
        <div className="grid gap-6 lg:grid-cols-[auto_1fr_auto] lg:items-center">
          <div
            className="shrink-0 flex items-center justify-center rounded-full"
            style={{
              width: '5rem',
              height: '5rem',
              background: 'var(--accent)',
              color: '#fff',
              fontSize: '1.6rem',
              fontWeight: 800,
              letterSpacing: '-0.02em',
            }}
          >
            {initials}
          </div>
          <div>
            <p className="section-label">{dict.profile.label}</p>
            <h1 className="mt-2" style={{ fontSize: 'clamp(1.5rem, 2.4vw, 2.2rem)', fontWeight: 800, color: 'var(--ink-strong)', letterSpacing: '-0.03em' }}>
              {displayName}
            </h1>
            <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
              {user.email}
            </p>
            <p className="mt-1 text-xs" style={{ color: 'var(--ink-muted)' }}>
              {dict.profile.memberSince} {memberSince}
            </p>
          </div>
          <Link href="/upload" className="cta-primary" style={{ padding: '0.75rem 1.1rem', fontSize: '0.85rem' }}>
            {dict.profile.uploadNew}
          </Link>
        </div>
      </section>

      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        <div className="metric-tile">
          <p className="metric-value">{runs.length}</p>
          <p className="metric-label">{dict.profile.totalRuns}</p>
        </div>
        <div className="metric-tile">
          <p className="metric-value">{completedRuns.length}</p>
          <p className="metric-label">{dict.profile.completed}</p>
        </div>
        <div className="metric-tile">
          <p className="metric-value">{avgScore ?? '—'}</p>
          <p className="metric-label">{dict.profile.averageScore}</p>
        </div>
        <div className="metric-tile metric-tile--high">
          <div className="metric-tile-dot" style={{ background: 'var(--accent)' }} />
          <p className="metric-value" style={{ color: 'var(--accent)' }}>{bestScore ?? '—'}</p>
          <p className="metric-label">{dict.profile.bestScore}</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <ScoreTrendCard
          runs={progressScoredRuns}
          title={dict.profile.trendTitle}
          subtitle={dict.profile.trendSubtitle}
          lang={lang}
        />

        <section className="surface-card p-6">
          <p className="section-label">{dict.profile.summary}</p>
          <div className="mt-4 space-y-4">
            <div className="surface-card-muted p-4">
              <p className="text-xs font-bold uppercase tracking-[0.12em]" style={{ color: 'var(--ink-muted)' }}>
                {dict.profile.latestRun}
              </p>
              <p className="mt-2 text-base font-semibold" style={{ color: 'var(--ink-strong)' }}>
                {latestCompleted ? getJobDisplayName(latestCompleted) : dict.profile.noCompleted}
              </p>
              {latestCompleted && (
                <div className="mt-2 space-y-3">
                  <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                    {latestCompletedExcluded
                      ? dict.profile.latestExcluded.replace('{score}', String(latestCompleted.score))
                      : latestCompleted.score != null
                      ? dict.profile.latestScored.replace('{score}', String(latestCompleted.score))
                      : dict.profile.latestReview}
                  </p>
                  <RunMetadataEditor
                    jobId={latestCompleted.id}
                    initialDisplayName={getJobDisplayName(latestCompleted)}
                    initialUserNote={getJobUserNote(latestCompleted)}
                  />
                </div>
              )}
            </div>
            <div className="surface-card-muted p-4">
              <p className="text-xs font-bold uppercase tracking-[0.12em]" style={{ color: 'var(--ink-muted)' }}>
                {dict.profile.bestRun}
              </p>
              <p className="mt-2 text-base font-semibold" style={{ color: 'var(--ink-strong)' }}>
                {bestRun ? getJobDisplayName(bestRun) : dict.profile.noScored}
              </p>
              {bestRun && (
                <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  {dict.profile.peakScore} {bestRun.score} · {translateKnownText(scoreLabel(bestRun.score), lang)}
                </p>
              )}
            </div>
          </div>
        </section>
      </div>

      <section className="surface-card p-6 lg:p-8">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <p className="section-label">{dict.profile.runHistory}</p>
            <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
              {dict.profile.recentSessions}
            </h2>
          </div>
          <Link href="/jobs" className="cta-secondary" style={{ padding: '0.6rem 1rem', fontSize: '0.82rem' }}>
            {dict.profile.openArchive}
          </Link>
        </div>

        {!recentRuns.length ? (
          <div className="mt-5 surface-card-muted p-8 text-center">
            <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>
              {dict.profile.noRuns}
            </p>
          </div>
        ) : (
          <ul className="mt-5 space-y-3">
            {recentRuns.map((job) => {
              const presentation = resolveJobPresentation(job)
              const previewUrl = previewUrlByJob.get(job.id) ?? null
              const displayNameForRun = getJobDisplayName(job)
              const originalFilename = getJobOriginalFilename(job)
              const userNote = getJobUserNote(job)
              const excludedFromProgress = job.score != null && !scoreCountsForProgress(summaryByJob.get(job.id))
              return (
                <li key={job.id}>
                  <div className="surface-card-muted flex items-center gap-4 px-4 py-4">
                    <Link href={`/jobs/${job.id}`} className="flex min-w-0 flex-1 items-center gap-4">
                      {previewUrl ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={previewUrl}
                          alt={displayNameForRun}
                          className="rounded-[var(--radius-lg)] shrink-0"
                          style={{ width: '4.5rem', height: '4.5rem', objectFit: 'cover' }}
                        />
                      ) : (
                        <div
                          className="rounded-[var(--radius-lg)] flex items-center justify-center shrink-0"
                          style={{ width: '4.5rem', height: '4.5rem', background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.06)' }}
                        >
                          <div className="w-2.5 h-2.5 rounded-full" style={{ background: presentation.dot }} />
                        </div>
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-sm font-semibold truncate" style={{ color: 'var(--ink-strong)' }}>
                            {displayNameForRun}
                          </p>
                          <span
                            className="text-xs font-semibold px-2.5 py-1 rounded-full"
                            style={{ color: presentation.dot, background: presentation.pill }}
                          >
                            {translateKnownText(presentation.label, lang)}
                          </span>
                        </div>
                        <p className="mt-1 text-xs" style={{ color: 'var(--ink-muted)' }}>
                          {formatDateTime(job.created_at, lang, { dateStyle: 'medium', timeStyle: 'short' })}
                          {originalFilename && originalFilename !== displayNameForRun ? ` · ${originalFilename}` : ''}
                        </p>
                        {userNote && (
                          <p className="mt-1 text-xs" style={{ color: 'var(--ink-soft)' }}>
                            {userNote}
                          </p>
                        )}
                      </div>
                    </Link>

                    <div className="flex items-center gap-3 shrink-0">
                      {job.score != null && (
                        <div className="text-right">
                          <p className="text-base font-bold" style={{ color: 'var(--accent)' }}>
                            {job.score}
                          </p>
                          <p className="text-xs" style={{ color: 'var(--ink-muted)' }}>
                            {excludedFromProgress
                              ? dict.profile.excludedFromProgress
                              : translateKnownText(scoreLabel(job.score), lang)}
                          </p>
                        </div>
                      )}
                      <JobRetryAction
                        jobId={job.id}
                        canRetry={presentation.canRetry}
                        actionLabel={presentation.actionLabel}
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
  )
}
