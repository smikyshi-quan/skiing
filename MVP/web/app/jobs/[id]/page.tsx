'use client'

import { useEffect, useState } from 'react'
import { useParams, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { buildTechniqueDashboard, scoreLabel, computeReliability, buildReliabilityMessage, generateLimitations, isJudgeUnavailable, scoreCountsForProgress, type TechniqueRunSummary, type RecapReliability, type AiCoaching, type AiCoachingPoint } from '@/lib/analysis-summary'
import { getDrill, localizeDrill } from '@/lib/drills'
import type { ArtifactWithUrl, Job, JobStatus } from '@/lib/types'
import { getJobDisplayName, getJobUserNote } from '@/lib/job-ui'
import { RunMetadataEditor } from '@/components/run-metadata-editor'
import { JobRetryAction } from '@/components/job-retry-action'
import { useLanguage } from '@/components/language-provider'
import { createClient as createBrowserClient } from '@/lib/supabase/client'
import { formatDateTime, reliabilityCopy, scoreContextForLang, translateKnownText } from '@/lib/i18n'

interface JobPresentationResponse {
  label: string
  helper: string
  dot: string
  pill: string
  canRetry: boolean
  actionLabel: string | null
}

interface JobResponse {
  job: Job
  artifacts: ArtifactWithUrl[]
  summary: TechniqueRunSummary | null
  previousScore: number | null
  aiCoaching: AiCoaching | null
  presentation?: JobPresentationResponse
}

type Tab = 'recap' | 'metrics' | 'moments' | 'downloads'

const ACTIVE: Set<JobStatus> = new Set(['created', 'uploaded', 'queued', 'running'])

const STATUS_META: Record<JobStatus, { label: string; color: string; background: string; helper: string }> = {
  created: { label: 'Preparing upload', color: 'var(--ink-soft)', background: 'rgba(17,17,17,0.04)', helper: 'We are setting up your run.' },
  uploaded: { label: 'Upload complete', color: 'var(--accent)', background: 'var(--accent-dim)', helper: 'Your video is ready. Analysis usually starts within a minute.' },
  queued: { label: 'Queued for analysis', color: 'var(--gold)', background: 'var(--gold-dim)', helper: 'We are waiting for a worker slot. Most runs finish in 1-2 minutes.' },
  running: { label: 'Analyzing run', color: 'var(--accent)', background: 'var(--accent-dim)', helper: 'We are reviewing your technique and writing the recap. Most runs finish in 1-2 minutes.' },
  done: { label: 'Recap ready', color: 'var(--success)', background: 'var(--success-dim)', helper: 'Your feedback is ready to review.' },
  error: { label: 'Analysis failed', color: 'var(--danger)', background: 'var(--danger-dim)', helper: 'Retry with a cleaner single-run clip.' },
}

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'recap', label: 'Recap' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'moments', label: 'Moments' },
  { id: 'downloads', label: 'Downloads' },
]

const CATEGORY_COLORS: Record<string, { accent: string; badge: string }> = {
  balance: { accent: 'coaching-accent-balance', badge: 'category-badge-balance' },
  edging: { accent: 'coaching-accent-edging', badge: 'category-badge-edging' },
  rhythm: { accent: 'coaching-accent-rhythm', badge: 'category-badge-rhythm' },
  movement: { accent: 'coaching-accent-movement', badge: 'category-badge-movement' },
  general: { accent: 'coaching-accent-general', badge: 'category-badge-general' },
}

function signedDownloads(artifacts: ArtifactWithUrl[]) {
  return [
    { label: 'Annotated recap video', artifact: artifacts.find((a) => a.kind === 'video_overlay') },
  ].filter((e): e is { label: string; artifact: ArtifactWithUrl } => Boolean(e.artifact?.url))
}

function confidenceMeta(
  reliability: RecapReliability,
  reliabilityMessage: ReturnType<typeof buildReliabilityMessage> | null,
  lang: 'en' | 'zh',
) {
  const localized = reliabilityCopy(reliability, lang)

  if (reliability === 'insufficient') {
    return {
      label: localized.badge,
      color: 'var(--gold)',
      background: 'var(--gold-dim)',
      helper: localized.helper,
    }
  }

  if (reliability === 'limited') {
    return {
      label: localized.badge,
      color: 'var(--gold)',
      background: 'var(--gold-dim)',
      helper: localized.helper,
    }
  }

  return {
    label: localized.badge,
    color: 'var(--success)',
    background: 'var(--success-dim)',
    helper: localized.helper,
  }
}

function metricDotColor(value: number, threshold: number): string {
  return value >= threshold ? 'var(--accent)' : 'var(--gold)'
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function progressWindow(job: Job) {
  const step = typeof job.config?.progress_step === 'number' ? job.config.progress_step : null
  const total = typeof job.config?.progress_total === 'number' ? job.config.progress_total : null

  if (job.status === 'done') return { floor: 100, cap: 100 }
  if (job.status === 'error') return { floor: 100, cap: 100 }

  if (step != null && total && total > 0) {
    const safeStep = clamp(step, 1, total)
    const floor = clamp(Math.round(((safeStep - 1) / total) * 100) + 6, 6, 94)
    const cap = safeStep === total
      ? 96
      : clamp(Math.round((safeStep / total) * 100) + 10, floor + 8, 96)
    return { floor, cap }
  }

  switch (job.status) {
    case 'created':
      return { floor: 8, cap: 18 }
    case 'uploaded':
      return { floor: 16, cap: 30 }
    case 'queued':
      return { floor: 24, cap: 42 }
    case 'running':
      return { floor: 40, cap: 92 }
    default:
      return { floor: 8, cap: 16 }
  }
}

export default function JobDetailPage() {
  const { lang, dict } = useLanguage()
  const { id } = useParams<{ id: string }>()
  const searchParams = useSearchParams()

  const [data, setData] = useState<JobResponse | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('recap')
  const [displayProgress, setDisplayProgress] = useState(0)
  const [isAnonymous, setIsAnonymous] = useState(false)

  useEffect(() => {
    createBrowserClient().auth.getUser().then(({ data: { user } }) => {
      if (user?.is_anonymous) setIsAnonymous(true)
    })
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    async function loadJob() {
      try {
        const response = await fetch(`/api/jobs/${id}`)
        if (!response.ok) {
          throw new Error(response.status === 404 ? 'This run could not be found.' : 'We could not load this run right now.')
        }
        const json: JobResponse = await response.json()
        if (!cancelled) {
          setData(json)
          setFetchError(null)
        }
        return json.job.status
      } catch (error) {
        if (!cancelled) setFetchError(error instanceof Error ? error.message : 'We could not load this run right now.')
        return null
      }
    }

    async function poll() {
      const status = await loadJob()
      if (!cancelled && status && ACTIVE.has(status)) {
        timer = setTimeout(poll, 4000)
      }
    }

    poll()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [id])

  useEffect(() => {
    if (!data) {
      setDisplayProgress(0)
      return
    }

    const job = data.job
    const isActive = ACTIVE.has(job.status)
    if (!isActive) {
      setDisplayProgress(job.status === 'done' ? 100 : 0)
      return
    }

    const { floor, cap } = progressWindow(job)
    setDisplayProgress((prev) => {
      const seeded = prev > 0 ? prev : floor
      return clamp(Math.max(seeded, floor), floor, cap)
    })

    const timer = setInterval(() => {
      setDisplayProgress((prev) => {
        const current = clamp(Math.max(prev, floor), floor, cap)
        if (current >= cap) return current
        const next = current + Math.max(0.8, (cap - current) * 0.12)
        return clamp(Math.round(next * 10) / 10, floor, cap)
      })
    }, 900)

    return () => clearInterval(timer)
  }, [data])

  if (fetchError && !data) {
    return (
      <>
        <div className="space-y-4">
          <div className="surface-card-strong p-6" style={{ color: 'var(--danger)', background: 'var(--danger-dim)' }}>
            {fetchError}
          </div>
          <Link href="/jobs" className="cta-secondary">{dict.job.backToArchive}</Link>
        </div>
      </>
    )
  }

  if (!data) {
    return (
      <>
        <div className="space-y-4 animate-pulse">
          <div className="h-6 w-36 rounded-full" style={{ background: 'rgba(0,0,0,0.08)' }} />
          <div className="surface-card h-[24rem]" />
        </div>
      </>
    )
  }

  const { job, artifacts, summary, previousScore, aiCoaching } = data
  const statusMeta = STATUS_META[job.status]
  const dashboard = summary ? buildTechniqueDashboard(summary) : null
  const reliability: RecapReliability = dashboard?.reliability ?? (summary ? computeReliability(summary) : 'reliable')
  const reliabilityMessage = summary ? buildReliabilityMessage(summary) : null
  const confidence = confidenceMeta(reliability, reliabilityMessage, lang)
  const reliabilityUi = reliabilityCopy(reliability, lang)
  const isActive = ACTIVE.has(job.status)
  const fromUpload = searchParams.get('fromUpload') === '1'
  const progressNote = typeof job.config?.progress_note === 'string' ? job.config.progress_note : null
  const overlayArtifact = artifacts.find((artifact) => artifact.kind === 'video_overlay')
  const coolMomentPhotos = artifacts.filter((artifact) => artifact.kind === 'cool_moment_photo')
  const peakFrames = artifacts.filter((artifact) => artifact.kind === 'peak_pressure_frame' || artifact.kind === 'peak_pressure_frame_enhanced')
  const downloads = signedDownloads(artifacts)
  const judgeUnavailable = isJudgeUnavailable(aiCoaching)
  const showJudgeUnavailable = job.status === 'done' && (!aiCoaching || judgeUnavailable)
  const currentCountsForProgress = summary ? scoreCountsForProgress(summary) : true
  const score = job.score ?? dashboard?.overview.overallScore ?? null
  const level = score != null ? scoreLabel(score) : null
  // Hero headline is the qualitative level for the score beside it. The judge's
  // top coaching point is not used here — it already leads the 复盘 section.
  const heroTitle = level
    ? reliability !== 'reliable'
      ? `${translateKnownText(level, lang)}${lang === 'zh' ? '（参考）' : ' (tentative)'}`
      : translateKnownText(level, lang)
    : dict.job.reviewMovement
  const scoreDelta = score != null && previousScore != null && currentCountsForProgress ? score - previousScore : null
  const displayName = getJobDisplayName(job)
  const userNote = getJobUserNote(job)
  const breadcrumbName = displayName
  const presentation = data.presentation

  return (
    <>
      <div className="space-y-6">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 px-1 sm:px-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
          <Link href="/jobs" className="hover:underline">{dict.job.archive}</Link>
          <span>/</span>
          <span className="font-mono" style={{ color: 'var(--ink-strong)' }}>{breadcrumbName}</span>
        </div>

        {fromUpload && isActive && (
          <section className="surface-card p-4" style={{ background: 'rgba(0,132,212,0.06)', border: '1px solid rgba(0,132,212,0.15)' }}>
            <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
              {dict.job.uploadBannerTitle}
            </p>
            <p className="mt-1 text-sm" style={{ color: 'var(--ink-soft)' }}>
              {dict.job.uploadBannerBody}
            </p>
          </section>
        )}

        {/* ── Run Recap hero ──────────────────────────── */}
        <section className="surface-card p-6 lg:p-8">
          <div className="grid gap-6 lg:grid-cols-[1.06fr_0.94fr]">
            {/* Left: score + headline */}
            <div className="space-y-5">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="eyebrow">{dict.job.runRecap}</span>
                  <span
                    className="status-pill"
                    style={{ color: confidence.color, background: confidence.background }}
                  >
                    {confidence.label}
                  </span>
                  {reliability !== 'reliable' && (
                    <span
                      className="text-xs font-bold px-2.5 py-1 rounded-full"
                      style={{ color: 'var(--ink-soft)', background: 'rgba(17,17,17,0.04)', border: '1px solid rgba(17,17,17,0.08)' }}
                    >
                      {dict.job.reviewConfidence}
                    </span>
                  )}
                </div>
                <span className="status-pill" style={{ color: presentation?.dot ?? statusMeta.color, background: presentation?.pill ?? statusMeta.background }}>
                  {translateKnownText(presentation?.label ?? statusMeta.label, lang)}
                </span>
              </div>

              {/* Score + headline row */}
              <div className="flex items-start gap-5">
                <div className="shrink-0">
                  <p className="text-xs font-bold uppercase tracking-[0.18em]" style={{ color: 'var(--ink-muted)' }}>
                    {dict.job.currentScore}
                  </p>
                  {score != null ? (
                    <div className="score-ring mt-2" style={{ width: '8.8rem', height: '8.8rem' }}>
                      <div className="score-ring-glow" />
                      <svg width="140" height="140" viewBox="0 0 140 140">
                        <circle cx="70" cy="70" r="54" fill="none" stroke="rgba(17,17,17,0.08)" strokeWidth="7" />
                        <circle
                          cx="70" cy="70" r="54"
                          fill="none"
                          stroke="url(#scoreGradDetail)"
                          strokeWidth="7"
                          strokeLinecap="round"
                          strokeDasharray="339.29"
                          strokeDashoffset={339.29 - (score / 100) * 339.29}
                        />
                        <defs>
                          <linearGradient id="scoreGradDetail" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stopColor="#0084d4" />
                            <stop offset="100%" stopColor="#c79a44" />
                          </linearGradient>
                        </defs>
                      </svg>
                      <div className="score-ring-label">
                        <span className="font-extrabold tracking-tight" style={{ fontSize: '2.15rem', color: 'var(--ink-strong)' }}>
                          {score}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div
                      className="mt-2 flex items-center justify-center rounded-full"
                      style={{ width: '8.8rem', height: '8.8rem', background: 'rgba(255,255,255,0.9)', border: '2px dashed rgba(17,17,17,0.12)' }}
                    >
                      <div className="text-center px-2">
                        <p className="text-sm font-bold" style={{ color: 'var(--ink-soft)' }}>{dict.job.noScore}</p>
                        <p className="mt-1 text-xs" style={{ color: 'var(--ink-muted)' }}>{dict.job.forThisClip}</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <h1 style={{ fontSize: 'clamp(1.9rem, 3.6vw, 3rem)', fontWeight: 800, letterSpacing: '-0.05em', lineHeight: 1.05, color: 'var(--ink-strong)' }}>
                    {heroTitle}
                  </h1>
                  <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-base)', maxWidth: '40rem' }}>
                    {confidence.helper}
                  </p>
                  <div className="mt-3 flex items-center gap-2 flex-wrap">
                    {scoreDelta != null && (
                      <span
                        className="text-xs font-bold px-2 py-0.5 rounded-full"
                        style={{
                          color: scoreDelta >= 0 ? 'var(--success)' : 'var(--danger)',
                          background: scoreDelta >= 0 ? 'var(--success-dim)' : 'var(--danger-dim)',
                        }}
                      >
                        {scoreDelta >= 0 ? '+' : ''}{scoreDelta} {dict.job.vsPrev}
                      </span>
                    )}
                    {score != null && (
                      <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                        {scoreContextForLang(score, lang)}
                      </span>
                    )}
                    {score != null && !currentCountsForProgress && (
                      <span className="text-xs font-bold px-2 py-0.5 rounded-full" style={{ color: 'var(--gold)', background: 'var(--gold-dim)' }}>
                        {dict.job.excludedFromProgress}
                      </span>
                    )}
                  </div>
                  {reliability === 'insufficient' && reliabilityMessage && (
                    <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-soft)' }}>
                      {reliabilityUi.explanation}
                    </p>
                  )}
                  {userNote && (
                    <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                      {dict.job.notePrefix} {userNote}
                    </p>
                  )}
                </div>
              </div>

              {/* Video */}
              {overlayArtifact?.url ? (
                <div
                  className="overflow-hidden"
                  style={{ borderRadius: 'var(--radius-xl)', background: '#0a0f1a', border: '1px solid rgba(17,17,17,0.08)' }}
                >
                  <video src={overlayArtifact.url} controls playsInline className="w-full aspect-video bg-black" />
                </div>
              ) : (
                <div
                  className="aspect-video flex items-center justify-center text-center p-8"
                  style={{ borderRadius: 'var(--radius-xl)', background: 'rgba(255,255,255,0.9)', border: '1px solid rgba(17,17,17,0.08)', color: 'var(--ink-soft)' }}
                >
                  {dict.job.videoPending}
                </div>
              )}
            </div>

            {/* Right sidebar: metrics + context */}
            <aside className="space-y-4">
              {isActive && (() => {
                const stage = typeof job.config?.progress_stage === 'string' ? job.config.progress_stage : null
                const label = stage
                  ? stage
                  : progressNote ?? statusMeta.helper
                return (
                  <div className="surface-card-muted p-5">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{dict.job.processing}</p>
                      <span className="text-xs" style={{ color: 'var(--ink-soft)' }}>{dict.job.autoRefresh}</span>
                    </div>
                    <div className="mt-3 progress-track">
                      <div className="progress-fill transition-all duration-700" style={{ width: `${displayProgress}%` }} />
                    </div>
                    <p className="mt-2 text-sm" style={{ color: 'var(--ink-base)' }}>{translateKnownText(label, lang)}</p>
                    <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>{dict.job.processingEta}</p>
                  </div>
                )
              })()}

              {/* Quick metrics */}
              <div className="surface-card-muted p-5">
                <p className="section-label">{dict.job.techniqueSummary}</p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className={dashboard && dashboard.overview.overallScore > 60 ? 'metric-tile metric-tile--high' : dashboard ? 'metric-tile metric-tile--low' : 'metric-tile'}>
                    <div className="metric-tile-dot" style={{ background: dashboard ? metricDotColor(dashboard.overview.overallScore, 60) : 'var(--ink-muted)' }} />
                    <p className="metric-value" style={{ color: dashboard && dashboard.overview.overallScore > 60 ? 'var(--accent)' : 'var(--gold)' }}>
                      {dashboard ? dashboard.overview.overallScore : '—'}
                    </p>
                    <p className="metric-label">{dict.sample.techniqueScore}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard ? dashboard.overview.turnsDetected : artifacts.length}</p>
                    <p className="metric-label">{dashboard ? dict.sample.turnsDetected : dict.job.filesReady}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard ? `${dashboard.overview.edgeAngle.toFixed(0)}°` : '—'}</p>
                    <p className="metric-label">{dict.job.edgeAngle}</p>
                  </div>
                  <div className={dashboard && reliability === 'reliable' ? 'metric-tile metric-tile--high' : dashboard ? 'metric-tile metric-tile--low' : 'metric-tile'}>
                    <div className="metric-tile-dot" style={{ background: dashboard && reliability === 'reliable' ? 'var(--accent)' : 'var(--gold)' }} />
                    <p className="metric-value" style={{ color: dashboard && reliability === 'reliable' ? 'var(--accent)' : 'var(--gold)' }}>
                      {dashboard ? translateKnownText(dashboard.overview.clipQualityLabel, lang) : '—'}
                    </p>
                    <p className="metric-label">{dict.job.clipQuality}</p>
                  </div>
                </div>
              </div>

              {/* Run context */}
              <div className="surface-card-muted p-5">
                <p className="section-label">{dict.job.runContext}</p>
                <div className="mt-3 space-y-2 text-sm" style={{ color: 'var(--ink-base)' }}>
                  <p>
                    <span style={{ color: 'var(--ink-muted)' }}>{dict.job.runTitle}:</span>{' '}
                    {displayName}
                  </p>
                  <p>
                    <span style={{ color: 'var(--ink-muted)' }}>{dict.job.uploaded}:</span>{' '}
                    {formatDateTime(job.created_at, lang)}
                  </p>
                  <p>
                    <span style={{ color: 'var(--ink-muted)' }}>{dict.job.updated}:</span>{' '}
                    {formatDateTime(job.updated_at, lang)}
                  </p>
                  <p>
                    <span style={{ color: 'var(--ink-muted)' }}>{dict.job.status}:</span>{' '}
                    {translateKnownText(presentation?.helper ?? progressNote ?? statusMeta.helper, lang)}
                  </p>
                </div>
              </div>

              <div className="surface-card-muted p-5">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label">{dict.job.runDetails}</p>
                    <p className="mt-2 text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
                      {dict.job.runDetailsBody}
                    </p>
                  </div>
                  <JobRetryAction
                    jobId={job.id}
                    canRetry={presentation?.canRetry ?? false}
                    actionLabel={presentation?.actionLabel ?? null}
                  />
                </div>
                <div className="mt-4">
                  <RunMetadataEditor
                    jobId={job.id}
                    initialDisplayName={displayName}
                    initialUserNote={userNote}
                  />
                </div>
              </div>

              {job.error && (
                <div className="surface-card-muted p-5 text-sm" style={{ color: 'var(--danger)', background: 'var(--danger-dim)' }}>
                  {job.error}
                </div>
              )}
            </aside>
          </div>
        </section>

        {/* ── Tab navigation ──────────────────────────── */}
        <section className="surface-card-strong p-3 flex flex-wrap gap-2" style={{ position: 'sticky', top: 'var(--sticky-tabs-offset)', zIndex: 30 }}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className="rounded-full px-4 py-2 text-sm font-semibold transition-colors"
              style={{
                background: activeTab === tab.id ? 'rgba(0,0,0,0.06)' : 'transparent',
                color: activeTab === tab.id ? 'var(--ink-strong)' : 'var(--ink-soft)',
                border: activeTab === tab.id ? '1px solid rgba(0,0,0,0.06)' : '1px solid transparent',
              }}
            >
              {tab.id === 'recap'
                ? dict.job.recap
                : tab.id === 'metrics'
                  ? dict.job.metrics
                  : tab.id === 'moments'
                    ? dict.job.moments
                    : dict.job.downloads}
            </button>
          ))}
        </section>

        {/* ── Recap tab ───────────────────────────────── */}
        {activeTab === 'recap' && (
          <div className="space-y-6">
            {/* Reliability banners */}
            {reliability !== 'reliable' && reliabilityMessage && (
              <div className="surface-card p-5 flex items-start gap-3" style={{
                background: reliability === 'insufficient' ? 'var(--gold-dim)' : 'rgba(0,132,212,0.06)',
                border: reliability === 'insufficient' ? '1px solid rgba(199,154,68,0.25)' : '1px solid rgba(0,132,212,0.15)',
              }}>
                  <span style={{ fontSize: '1.25rem' }}>{reliability === 'insufficient' ? '\u26A0' : '\u2139'}</span>
                <div>
                  <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{reliabilityUi.title}</p>
                  <p className="mt-1 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                    {reliabilityUi.explanation}
                  </p>
                  <p className="mt-2 text-sm leading-6" style={{ color: 'var(--ink-soft)' }}>
                    {reliabilityUi.nextStep}
                  </p>
                </div>
              </div>
            )}

            {/* ── Coach's Analysis ─────────────────────── */}
            {aiCoaching && !judgeUnavailable ? (
              <section className="surface-card p-6">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label" style={{ color: 'var(--accent)' }}>{dict.job.coachAnalysis}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.job.coachAnalysisTitle}
                    </h2>
                  </div>
                  <span className="eyebrow">{dict.job.aiCoach}</span>
                </div>

                {/* Coach summary */}
                <div className="mt-5 coach-summary">
                  <p className="text-base leading-7" style={{ color: 'var(--ink-base)' }}>
                    {aiCoaching.coach_summary}
                  </p>
                </div>

                {/* Coaching points */}
                <div className="mt-5 space-y-3">
                  {aiCoaching.coaching_points.map((point: AiCoachingPoint, idx: number) => {
                    const catColors = CATEGORY_COLORS[point.category] ?? CATEGORY_COLORS.general
                    const CATEGORY_LABELS: Record<string, string> = {
                      movement: 'Movement', edging: 'Edging', rhythm: 'Rhythm', balance: 'Balance', general: 'General',
                    }
                    const drill = point.recommended_drill_id ? getDrill(point.recommended_drill_id) : null
                    const localizedDrill = drill ? localizeDrill(drill, lang) : null
                    return (
                      <div key={`${point.title}-${idx}`} className={`coaching-card ${catColors.accent}`}>
                        <div className="flex items-center gap-3 pl-3">
                          <span className="preflight-number" style={{ width: '1.5rem', height: '1.5rem', fontSize: '0.62rem' }}>
                            {String(idx + 1).padStart(2, '0')}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{point.title}</p>
                          </div>
                          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 ${catColors.badge}`}>
                            {translateKnownText(CATEGORY_LABELS[point.category] ?? point.category, lang)}
                          </span>
                        </div>
                        <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-base)' }}>
                          {point.feedback}
                        </p>
                        {localizedDrill && (
                          <div className="mt-3 pl-3">
                            <div className="drill-card inline-flex items-center gap-3 px-4 py-3">
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>{dict.job.recommendedDrill}</p>
                                <p className="mt-1 text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{localizedDrill.title}</p>
                                <p className="mt-1 text-xs" style={{ color: 'var(--ink-soft)' }}>{localizedDrill.description}</p>
                              </div>
                              {localizedDrill.videoUrl && (
                                <a
                                  href={localizedDrill.videoUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="cta-primary shrink-0"
                                  style={{ padding: '0.5rem 1rem', fontSize: '0.78rem' }}
                                >
                                  {dict.job.watchDrill}
                                </a>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>

                {/* Additional observations */}
                {aiCoaching.additional_observations?.length > 0 && (
                  <div className="mt-5">
                    <p className="section-label">{dict.job.additional}</p>
                    <ul className="mt-3 space-y-2">
                      {aiCoaching.additional_observations.map((obs: string, idx: number) => (
                        <li key={idx} className="text-sm leading-6 pl-3" style={{ color: 'var(--ink-base)', borderLeft: '2px solid rgba(0,0,0,0.08)', paddingLeft: '0.75rem' }}>
                          {obs}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            ) : showJudgeUnavailable ? (
              <section className="surface-card p-6">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label" style={{ color: 'var(--accent)' }}>{dict.job.coachAnalysis}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.job.judgeUnavailableTitle}
                    </h2>
                  </div>
                  <span className="eyebrow">{dict.job.aiCoach}</span>
                </div>
                <p className="mt-5 text-base leading-7" style={{ color: 'var(--ink-base)' }}>
                  {dict.job.judgeUnavailableBody}
                </p>
              </section>
            ) : (
              <section className="surface-card p-6">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label" style={{ color: 'var(--accent)' }}>{dict.job.coachAnalysis}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {isActive ? dict.job.coachWriting : dict.job.coachNotReady}
                    </h2>
                  </div>
                  <span className="eyebrow">{dict.job.aiCoach}</span>
                </div>
                <p className="mt-5 text-base leading-7" style={{ color: 'var(--ink-base)' }}>
                  {isActive
                    ? dict.job.coachWritingBody
                    : dict.job.coachMissingBody}
                </p>
              </section>
            )}

            {/* ── Recommended Practice (drill summary) ─── */}
            {aiCoaching && !judgeUnavailable && (() => {
              const recommendedDrills = aiCoaching.coaching_points
                .map((p: AiCoachingPoint) => p.recommended_drill_id)
                .filter((id): id is string => id != null)
                .map((id) => getDrill(id))
                .filter((d): d is NonNullable<typeof d> => d != null)
              const uniqueDrills = [...new Map(recommendedDrills.map((d) => [d.id, d])).values()]
              if (!uniqueDrills.length) return null
              return (
                <section className="surface-card p-6">
                  <p className="section-label">{dict.job.practice}</p>
                  <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                    {dict.job.practiceTitle}
                  </h2>
                  <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {uniqueDrills.map((drill) => {
                      const localizedDrill = localizeDrill(drill, lang)
                      return (
                      <div key={drill.id} className="drill-card p-4">
                        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                          CATEGORY_COLORS[drill.category]?.badge ?? 'category-badge-general'
                        }`}>
                          {translateKnownText(drill.category.charAt(0).toUpperCase() + drill.category.slice(1), lang)}
                        </span>
                        <h3 className="mt-3 text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{localizedDrill.title}</h3>
                        <p className="mt-1 text-xs leading-5" style={{ color: 'var(--ink-soft)' }}>{localizedDrill.description}</p>
                              {localizedDrill.videoUrl && (
                          <a
                            href={localizedDrill.videoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="cta-primary mt-3 w-full"
                            style={{ padding: '0.5rem 1rem', fontSize: '0.78rem' }}
                          >
                            {dict.job.watchDrill}
                          </a>
                        )}
                      </div>
                    )})}
                  </div>
                </section>
              )
            })()}

            {/* ── Model Limitations ────────────────────── */}
            {summary && (() => {
              const limitations = generateLimitations(summary)
              if (!limitations.length) return null
              return (
                <section className="surface-card p-6">
                  <p className="section-label" style={{ color: 'var(--gold)' }}>{dict.job.keepInMind}</p>
                  <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                    {dict.job.limitations}
                  </h2>
                  <div className="mt-5 space-y-3">
                    {limitations.map((lim, idx) => (
                      <div key={idx} className="limitations-card">
                        <h4>{translateKnownText(lim.title, lang)}</h4>
                        <p>{translateKnownText(lim.explanation, lang)}</p>
                      </div>
                    ))}
                  </div>
                </section>
              )
            })()}
          </div>
        )}

        {/* ── Metrics tab (Technique Markers) ─────────── */}
        {activeTab === 'metrics' && (
          <div className="space-y-6">
            <section className="grid gap-4 lg:grid-cols-2">
              {dashboard?.categories.map((category) => (
                <article key={category.id} className="surface-card p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="section-label">{translateKnownText(category.title, lang)}</p>
                      <p className="mt-2 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                        {lang === 'zh' ? '这些指标汇总了你这趟滑行里最关键的动作模式。' : 'These checks roll up the strongest movement patterns from your run.'}
                      </p>
                    </div>
                    <div className="text-center">
                      <div
                        className="w-14 h-14 rounded-full flex items-center justify-center text-lg font-extrabold"
                        style={{ background: 'var(--accent-dim)', color: 'var(--ink-strong)' }}
                      >
                        {category.score}
                      </div>
                      <p className="mt-1 text-xs font-bold uppercase tracking-[0.14em]" style={{ color: 'var(--ink-muted)' }}>
                        {translateKnownText(category.status, lang)}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 space-y-4">
                    {category.metrics.map((metric) => (
                      <div key={`${category.id}-${metric.label}`}>
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{translateKnownText(metric.label, lang)}</p>
                          <p className="font-mono text-xs" style={{ color: 'var(--accent)' }}>{metric.value}</p>
                        </div>
                        <p className="mt-1 text-sm" style={{ color: 'var(--ink-soft)' }}>{translateKnownText(metric.helper, lang)}</p>
                        <div className="mt-3 metric-rail">
                          <span style={{ width: `${metric.fill}%` }}>
                            <span className="metric-rail-dot" />
                          </span>
                        </div>
                        <div className="mt-1 flex items-center justify-between text-xs" style={{ color: 'var(--ink-muted)' }}>
                          <span>{translateKnownText(metric.leftLabel, lang)}</span>
                          <span>{translateKnownText(metric.rightLabel, lang)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </article>
              )) ?? (
                <article className="surface-card p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  {dict.job.metricsFallback}
                </article>
              )}
            </section>

            {!!dashboard?.turnHighlights.length && (
              <section className="surface-card p-6">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label">{dict.job.turnHighlights}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.job.bestTurns}
                    </h2>
                  </div>
                  <span className="status-pill" style={{ color: 'var(--success)', background: 'var(--success-dim)' }}>
                    {dict.job.techniqueScores}
                  </span>
                </div>

                <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                  {dashboard.turnHighlights.map((turn) => (
                    <div key={turn.title} className="surface-card-muted p-4">
                      <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{translateKnownText(turn.title, lang)}</p>
                      <p className="mt-3 text-3xl font-extrabold tracking-tight" style={{ color: 'var(--ink-strong)', fontVariantNumeric: 'tabular-nums' }}>
                        {turn.score}
                      </p>
                      <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>{translateKnownText(turn.detail, lang)}</p>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}

        {/* ── Moments tab ─────────────────────────────── */}
        {activeTab === 'moments' && (
          <div className="space-y-6">
            <section className="surface-card p-6">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label">{dict.job.keyMoments}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.job.strongestFrames}
                    </h2>
                  </div>
                  <span className="status-pill" style={{ color: 'var(--accent)', background: 'var(--accent-dim)' }}>
                    {coolMomentPhotos.length} {dict.job.photos}
                  </span>
                </div>

              {coolMomentPhotos.length ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {coolMomentPhotos.map((photo) => (
                    <a key={photo.id} href={photo.url} target="_blank" rel="noopener noreferrer" className="moment-card">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={photo.url} alt={translateKnownText(`Turn ${(photo.meta.turn_idx ?? 0) + 1}`, lang)} className="w-full aspect-[4/3] object-cover" />
                      <div className="moment-card-overlay">
                        <p className="text-xs font-mono text-white">
                          {translateKnownText(
                            `Turn ${(photo.meta.turn_idx ?? 0) + 1}${photo.meta.side ? ` · ${photo.meta.side}` : ''}${photo.meta.timestamp_s != null ? ` · ${Number(photo.meta.timestamp_s).toFixed(1)}s` : ''}`,
                            lang,
                          )}
                        </p>
                      </div>
                    </a>
                  ))}
                </div>
              ) : (
                <div className="mt-5 surface-card-muted p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  {dict.job.noMoments}
                </div>
              )}
            </section>

            <section className="surface-card p-6">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label">{dict.job.peakFrames}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.job.pressureSnapshots}
                    </h2>
                  </div>
                  <span className="status-pill" style={{ color: 'var(--gold)', background: 'var(--gold-dim)' }}>
                    {peakFrames.length} {dict.job.frames}
                  </span>
                </div>

              {peakFrames.length ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {peakFrames.map((frame) => (
                    <a key={frame.id} href={frame.url} target="_blank" rel="noopener noreferrer" className="moment-card">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={frame.url} alt={translateKnownText(`Turn ${(frame.meta.turn_idx ?? 0) + 1}`, lang)} className="w-full aspect-[4/3] object-cover" />
                      <div className="moment-card-overlay">
                        <p className="text-xs font-mono text-white">
                          {translateKnownText(
                            `Turn ${(frame.meta.turn_idx ?? 0) + 1}${frame.meta.side ? ` · ${frame.meta.side}` : ''}${frame.meta.timestamp_s != null ? ` · ${Number(frame.meta.timestamp_s).toFixed(1)}s` : ''}`,
                            lang,
                          )}
                        </p>
                      </div>
                    </a>
                  ))}
                </div>
              ) : (
                <div className="mt-5 surface-card-muted p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  {dict.job.noPeakFrames}
                </div>
              )}
            </section>
          </div>
        )}

        {/* ── Downloads tab ───────────────────────────── */}
        {activeTab === 'downloads' && (
          <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
            <section className="surface-card p-6">
              <p className="section-label">{dict.job.exports}</p>
              <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                {dict.job.filesKeep}
              </h2>

              <div className="mt-5 space-y-3">
                {downloads.length ? downloads.map(({ label, artifact }) => (
                  <a
                    key={`${label}-${artifact.id}`}
                    href={artifact.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="surface-card-muted px-4 py-4 flex items-center justify-between gap-4"
                  >
                    <div>
                      <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{translateKnownText(label, lang)}</p>
                      <p className="mt-1 text-xs" style={{ color: 'var(--ink-soft)' }}>
                        {dict.job.openFile}
                      </p>
                    </div>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--ink-soft)' }}>
                      <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
                    </svg>
                  </a>
                )) : (
                  <div className="surface-card-muted p-5 text-sm" style={{ color: 'var(--ink-soft)' }}>
                    {dict.job.noDownloads}
                  </div>
                )}
              </div>
            </section>

            <section className="surface-card p-6">
              <p className="section-label">{dict.job.assets}</p>
              <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                {dict.job.included}
              </h2>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <div className="metric-tile">
                  <p className="metric-value">{overlayArtifact ? 1 : 0}</p>
                  <p className="metric-label">{dict.job.videoRecap}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{downloads.length}</p>
                  <p className="metric-label">{dict.job.downloadsReady}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{coolMomentPhotos.length}</p>
                  <p className="metric-label">{dict.job.highlightPhotos}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{peakFrames.length}</p>
                  <p className="metric-label">{dict.job.actionStills}</p>
                </div>
              </div>

              {aiCoaching?.coach_summary && !judgeUnavailable ? (
                <div className="mt-6 surface-card-muted p-4">
                  <p className="section-label">{dict.job.coachNote}</p>
                  <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                    {aiCoaching.coach_summary}
                  </p>
                </div>
              ) : null}
            </section>
          </div>
        )}
        {isAnonymous && (
          <section
            className="surface-card p-5"
            style={{ background: 'rgba(0,132,212,0.06)', border: '1px solid rgba(0,132,212,0.15)' }}
          >
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div>
                <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
                  {dict.guest.bannerTitle}
                </p>
                <p className="mt-1 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  {dict.guest.bannerBody}
                </p>
              </div>
              <Link
                href="/signup"
                className="cta-primary shrink-0"
                style={{ padding: '0.6rem 1.1rem', fontSize: '0.85rem' }}
              >
                {dict.guest.bannerCta}
              </Link>
            </div>
          </section>
        )}
      </div>
    </>
  )
}
