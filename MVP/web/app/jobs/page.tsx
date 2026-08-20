import { createClient, createServiceClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'
import { Job } from '@/lib/types'
import { groupBySeason } from '@/lib/seasons'
import { scoreCountsForProgress } from '@/lib/analysis-summary'
import { backfillMissingScores, loadPreviewUrlsForJobIds, loadSummariesForJobIds, resolveJobPresentation } from '@/lib/server-job-data'
import { getJobDisplayName, getJobUserNote, getJobOriginalFilename, getJobSearchText } from '@/lib/job-ui'
import { RunMetadataEditor } from '@/components/run-metadata-editor'
import { ArchiveRunsClient, type ArchiveRunItem } from '@/components/archive-runs-client'
import { formatDate, getDictionary, getSessionTypeLabel, translateKnownText } from '@/lib/i18n'
import { readLanguage } from '@/lib/i18n-server'

export const dynamic = 'force-dynamic'

export default async function ArchivePage({
  searchParams,
}: {
  searchParams?: { edit?: string | string[] }
}) {
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

  const seasonGroups = groupBySeason(runs)
  const scoredRuns = completedRuns.filter((job): job is Job & { score: number } => job.score != null)
  const summaryByJob = await loadSummariesForJobIds(service, completedRuns.map((job) => job.id))
  const progressScoredRuns = scoredRuns.filter((job) => scoreCountsForProgress(summaryByJob.get(job.id)))
  const avgScore = progressScoredRuns.length
    ? Math.round(progressScoredRuns.reduce((sum, job) => sum + job.score, 0) / progressScoredRuns.length)
    : null
  const previewUrlByJob = await loadPreviewUrlsForJobIds(service, runs.map((run) => run.id))
  const initialEditJobId =
    typeof searchParams?.edit === 'string'
      ? searchParams.edit
      : Array.isArray(searchParams?.edit)
        ? searchParams.edit[0] ?? null
        : null
  const selectedRun = initialEditJobId
    ? runs.find((run) => run.id === initialEditJobId) ?? runs[0] ?? null
    : runs[0] ?? null

  const archiveRuns: ArchiveRunItem[] = runs.map((job) => {
    const date = new Date(job.created_at)
    const sessionType = getSessionTypeLabel(job.config?.session_type, lang)
    const displayName = getJobDisplayName(job)
    const presentation = resolveJobPresentation(job)

    return {
      id: job.id,
      created_at: job.created_at,
      status: job.status,
      statusLabel: translateKnownText(presentation.label, lang),
      statusHelper: translateKnownText(presentation.helper, lang),
      statusTone: presentation.tone,
      statusDot: presentation.dot,
      statusPill: presentation.pill,
      canRetry: presentation.retryable,
      actionLabel: translateKnownText(presentation.actionLabel, lang),
      displayName,
      originalFilename: getJobOriginalFilename(job),
      userNote: getJobUserNote(job),
      subtitle: `${formatDate(date, lang, { dateStyle: 'medium' })}${lang === 'en' ? ` at ${formatDate(date, lang, { hour: '2-digit', minute: '2-digit' })}` : ` ${formatDate(date, lang, { hour: '2-digit', minute: '2-digit' })}`}${sessionType ? ` · ${sessionType}` : ''}`,
      score: job.score,
      scoreCountsForProgress: scoreCountsForProgress(summaryByJob.get(job.id)),
      previewUrl: previewUrlByJob.get(job.id) ?? null,
      sessionType,
      searchText: getJobSearchText(job),
    }
  })

  return (
    <div className="space-y-6">
        <section className="surface-card p-8 lg:p-10">
          <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <span className="eyebrow">{dict.archive.eyebrow}</span>
              <h1 className="section-title mt-6">{dict.archive.title}</h1>
              <p className="section-copy mt-4 max-w-xl">
                {dict.archive.body}
              </p>

              <div className="mt-6 flex flex-wrap gap-3">
                <Link href="/upload" className="cta-primary">
                  {dict.archive.newRun}
                </Link>
                <Link href="/" className="cta-secondary">
                  {dict.archive.backHub}
                </Link>
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="metric-tile">
                  <p className="metric-value">{runs.length}</p>
                  <p className="metric-label">{dict.archive.totalRuns}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{completedRuns.length}</p>
                  <p className="metric-label">{dict.archive.completed}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{avgScore ?? '—'}</p>
                  <p className="metric-label">{dict.archive.averageScore}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{seasonGroups.length}</p>
                  <p className="metric-label">{seasonGroups.length === 1 ? dict.archive.seasonTrackedOne : dict.archive.seasonTrackedMany}</p>
                </div>
              </div>

              {selectedRun && (
                <div className="surface-card-muted p-5">
                  <p className="section-label">{dict.archive.editTitle}</p>
                  <p className="mt-2 text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
                    {getJobDisplayName(selectedRun)}
                  </p>
                  {getJobUserNote(selectedRun) && (
                    <p className="mt-1 text-xs" style={{ color: 'var(--ink-soft)' }}>
                      {getJobUserNote(selectedRun)}
                    </p>
                  )}
                  <div className="mt-4">
                    <RunMetadataEditor
                      jobId={selectedRun.id}
                      initialDisplayName={getJobDisplayName(selectedRun)}
                      initialUserNote={getJobUserNote(selectedRun)}
                      defaultEditing={Boolean(initialEditJobId)}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {!runs.length ? (
          <section className="surface-card p-6">
            <div className="surface-card-muted p-10 text-center">
              <div
                className="w-16 h-16 rounded-[var(--radius-lg)] flex items-center justify-center mx-auto mb-4"
                style={{ background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.06)' }}
              >
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--ink-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M12 8v4M12 16h.01"/>
                </svg>
              </div>
              <p className="text-base font-bold" style={{ color: 'var(--ink-strong)' }}>{dict.archive.emptyTitle}</p>
              <p className="text-sm mt-2" style={{ color: 'var(--ink-soft)' }}>
                {dict.archive.emptyBody}
              </p>
            </div>
          </section>
        ) : (
          <ArchiveRunsClient runs={archiveRuns} initialEditJobId={initialEditJobId} />
        )}
    </div>
  )
}
