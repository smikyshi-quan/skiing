'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { buildTechniqueDashboard, scoreLabel, type CoachingTip, type TechniqueRunSummary } from '@/lib/analysis-summary'
import type { ArtifactWithUrl, Job, JobStatus } from '@/lib/types'

interface JobResponse {
  job: Job
  artifacts: ArtifactWithUrl[]
  summary: TechniqueRunSummary | null
  previousScore: number | null
}

type Tab = 'recap' | 'metrics' | 'moments' | 'downloads'

const ACTIVE: Set<JobStatus> = new Set(['created', 'uploaded', 'queued', 'running'])

const STATUS_META: Record<JobStatus, { label: string; color: string; background: string; helper: string }> = {
  created: {
    label: 'Job created',
    color: 'var(--ink-soft)',
    background: 'rgba(255,255,255,0.06)',
    helper: 'Your upload slot is ready.',
  },
  uploaded: {
    label: 'Video uploaded',
    color: 'var(--accent)',
    background: 'var(--accent-dim)',
    helper: 'The worker can pick up the clip now.',
  },
  queued: {
    label: 'Waiting in queue',
    color: 'var(--gold)',
    background: 'var(--gold-dim)',
    helper: 'The recap deck is staged next.',
  },
  running: {
    label: 'Analysing run',
    color: 'var(--accent)',
    background: 'var(--accent-dim)',
    helper: 'Overlay video and summary artifacts are being prepared.',
  },
  done: {
    label: 'Analysis complete',
    color: 'var(--success)',
    background: 'var(--success-dim)',
    helper: 'Your recap deck is ready to review.',
  },
  error: {
    label: 'Analysis failed',
    color: 'var(--danger)',
    background: 'var(--danger-dim)',
    helper: 'Retry with a cleaner single-run clip.',
  },
}

const TIP_META: Record<CoachingTip['severity'], { label: string; color: string; background: string }> = {
  action: { label: 'Action', color: 'var(--gold)', background: 'var(--gold-dim)' },
  warn: { label: 'Watch', color: 'var(--accent)', background: 'var(--accent-dim)' },
  info: { label: 'Note', color: 'var(--ink-soft)', background: 'rgba(255,255,255,0.06)' },
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

function tipCategory(tip: CoachingTip): string {
  const text = `${tip.title} ${tip.explanation}`.toLowerCase()
  if (text.includes('rotat') || text.includes('upper body') || text.includes('quiet') || text.includes('counter')) return 'movement'
  if (text.includes('edge') || text.includes('carv') || text.includes('angulat') || text.includes('tilt')) return 'edging'
  if (text.includes('rhythm') || text.includes('tempo') || text.includes('timing') || text.includes('flow') || text.includes('pace')) return 'rhythm'
  if (text.includes('stance') || text.includes('balance') || text.includes('center') || text.includes('weight') || text.includes('narrow') || text.includes('wide')) return 'balance'
  return 'movement'
}

function signedDownloads(artifacts: ArtifactWithUrl[]) {
  return [
    { label: 'Overlay video', artifact: artifacts.find((artifact) => artifact.kind === 'video_overlay') },
    { label: 'Summary JSON', artifact: artifacts.find((artifact) => artifact.kind === 'summary_json') },
    { label: 'Metrics CSV', artifact: artifacts.find((artifact) => artifact.kind === 'metrics_csv') },
  ].filter((entry): entry is { label: string; artifact: ArtifactWithUrl } => Boolean(entry.artifact?.url))
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

function coachingHeadline(job: Job, summary: TechniqueRunSummary | null) {
  if (summary?.coaching_tips?.length) {
    return summary.coaching_tips[0].explanation
  }
  if (job.status === 'done') {
    return 'Great work getting out there. Review the overlay, key frames, and summary artifacts below.'
  }
  if (job.status === 'error') {
    return 'This run did not complete. A cleaner single-athlete clip usually gets the recap back on track.'
  }
  return 'Your run is progressing through the queue. The recap will refresh automatically.'
}

function metricTierClass(value: number, threshold: number): string {
  return value >= threshold ? 'metric-tile metric-tile--high' : 'metric-tile metric-tile--low'
}

function metricDotColor(value: number, threshold: number): string {
  return value >= threshold ? 'var(--accent)' : 'var(--gold)'
}

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>()

  const [data, setData] = useState<JobResponse | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('recap')

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    async function loadJob() {
      try {
        const response = await fetch(`/api/jobs/${id}`)
        if (!response.ok) throw new Error(`${response.status}`)
        const json: JobResponse = await response.json()
        if (!cancelled) {
          setData(json)
          setFetchError(null)
        }
        return json.job.status
      } catch (error) {
        if (!cancelled) {
          setFetchError(error instanceof Error ? error.message : 'Failed to load run')
        }
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

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [id])

  if (fetchError && !data) {
    return (
      <div className="space-y-4">
        <div
          className="surface-card-strong p-6"
          style={{ color: 'var(--danger)', background: 'var(--danger-dim)' }}
        >
          {fetchError}
        </div>
        <Link href="/jobs" className="cta-secondary">
          Back to archive
        </Link>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="h-6 w-36 rounded-full" style={{ background: 'rgba(255,255,255,0.08)' }} />
        <div className="surface-card h-[24rem]" />
      </div>
    )
  }

  const { job, artifacts, summary, previousScore } = data
  const statusMeta = STATUS_META[job.status]
  const dashboard = summary ? buildTechniqueDashboard(summary) : null
  const isActive = ACTIVE.has(job.status)
  const progressNote = typeof job.config?.progress_note === 'string' ? job.config.progress_note : null
  const overlayArtifact = artifacts.find((artifact) => artifact.kind === 'video_overlay')
  const coolMomentPhotos = artifacts.filter((artifact) => artifact.kind === 'cool_moment_photo')
  const peakFrames = artifacts.filter(
    (artifact) => artifact.kind === 'peak_pressure_frame' || artifact.kind === 'peak_pressure_frame_enhanced',
  )
  const downloads = signedDownloads(artifacts)
  const headline = coachingHeadline(job, summary)

  const score = job.score ?? dashboard?.overview.overallScore ?? null
  const level = score != null ? scoreLabel(score) : null
  const scoreDelta = score != null && previousScore != null ? score - previousScore : null

  const breadcrumbName =
    String(job.config?.original_filename ?? '') ||
    job.video_object_path?.split('/').pop() ||
    job.id.slice(0, 8)

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
        <Link href="/jobs" className="hover:underline">Archive</Link>
        <span>/</span>
        <span className="font-mono" style={{ color: 'var(--ink-strong)' }}>{breadcrumbName}</span>
      </div>

      {/* ── Score-first hero ─────────────────────────── */}
      <section className="surface-card p-6 lg:p-7">
        <div className="grid gap-6 lg:grid-cols-[1.16fr_0.84fr]">
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="space-y-3">
                <div className="flex items-center gap-4 flex-wrap">
                  {score != null && (
                    <div className="score-ring" style={{ width: '11.25rem', height: '11.25rem' }}>
                      <div className="score-ring-glow" />
                      <svg width="180" height="180" viewBox="0 0 180 180">
                        <circle cx="90" cy="90" r="76" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
                        <circle
                          cx="90" cy="90" r="76"
                          fill="none"
                          stroke="url(#scoreGradDetail)"
                          strokeWidth="8"
                          strokeLinecap="round"
                          strokeDasharray="477.52"
                          strokeDashoffset={477.52 - (score / 100) * 477.52}
                        />
                        <defs>
                          <linearGradient id="scoreGradDetail" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stopColor="#4fc3f7" />
                            <stop offset="100%" stopColor="#a78bfa" />
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
                        <span className="text-xs mt-1" style={{ color: 'var(--ink-soft)' }}>technique</span>
                      </div>
                    </div>
                  )}
                  <div className="space-y-2">
                    {level && (
                      <span className={levelBadgeClass(level)}>{level}</span>
                    )}
                    {scoreDelta != null && (
                      <span
                        className="text-sm font-bold px-2.5 py-1 rounded-full block w-fit"
                        style={{
                          color: scoreDelta >= 0 ? 'var(--success)' : 'var(--danger)',
                          background: scoreDelta >= 0 ? 'var(--success-dim)' : 'var(--danger-dim)',
                        }}
                      >
                        {scoreDelta >= 0 ? '+' : ''}{scoreDelta} vs prev
                      </span>
                    )}
                  </div>
                </div>
                <span className="eyebrow">Run recap</span>
                <h1 style={{ fontSize: 'clamp(1.8rem, 3vw, 2.8rem)', fontWeight: 700, letterSpacing: '-0.06em', color: 'var(--ink-strong)' }}>
                  {score != null ? headline.slice(0, 80) : 'Review how this run moved.'}
                </h1>
              </div>
              <span className="status-pill" style={{ color: statusMeta.color, background: statusMeta.background }}>
                {statusMeta.label}
              </span>
            </div>

            {overlayArtifact?.url ? (
              <div
                className="overflow-hidden rounded-[1.6rem]"
                style={{ background: '#0a0f1a', border: '1px solid rgba(255,255,255,0.07)' }}
              >
                <video src={overlayArtifact.url} controls playsInline className="w-full aspect-video bg-black" />
              </div>
            ) : (
              <div
                className="rounded-[1.6rem] aspect-video flex items-center justify-center text-center p-8"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', color: 'var(--ink-soft)' }}
              >
                Overlay video will appear here once the run is fully processed.
              </div>
            )}
          </div>

          <aside className="space-y-4">
            <div className="surface-card-muted p-5">
              <p className="text-xs uppercase tracking-[0.22em]" style={{ color: 'var(--ink-muted)' }}>
                Coaching headline
              </p>
              <p className="mt-3 text-base leading-7 font-medium" style={{ color: 'var(--ink-strong)' }}>
                {headline}
              </p>

              <div className="mt-5 grid grid-cols-2 gap-3">
                <div className={dashboard && dashboard.overview.overallScore > 60 ? 'metric-tile metric-tile--high' : dashboard ? 'metric-tile metric-tile--low' : 'metric-tile'}>
                  <div className="metric-tile-dot" style={{ background: dashboard ? metricDotColor(dashboard.overview.overallScore, 60) : 'var(--ink-muted)' }} />
                  <p className="metric-value" style={{ color: dashboard && dashboard.overview.overallScore > 60 ? 'var(--accent)' : 'var(--gold)' }}>
                    {dashboard ? dashboard.overview.overallScore : '—'}
                  </p>
                  <p className="metric-label">Technique score</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{dashboard ? dashboard.overview.turnsDetected : artifacts.length}</p>
                  <p className="metric-label">{dashboard ? 'Turns detected' : 'Artifacts ready'}</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{dashboard ? `${dashboard.overview.edgeAngle.toFixed(0)}°` : '—'}</p>
                  <p className="metric-label">Average edge angle</p>
                </div>
                <div className={dashboard && dashboard.overview.poseConfidence > 70 ? 'metric-tile metric-tile--high' : dashboard ? 'metric-tile metric-tile--low' : 'metric-tile'}>
                  <div className="metric-tile-dot" style={{ background: dashboard ? metricDotColor(dashboard.overview.poseConfidence, 70) : 'var(--ink-muted)' }} />
                  <p className="metric-value" style={{ color: dashboard && dashboard.overview.poseConfidence > 70 ? 'var(--accent)' : 'var(--gold)' }}>
                    {dashboard ? `${dashboard.overview.poseConfidence.toFixed(0)}%` : '—'}
                  </p>
                  <p className="metric-label">Pose confidence</p>
                </div>
              </div>
            </div>

            <div className="surface-card-muted p-5">
              <p className="text-xs uppercase tracking-[0.22em]" style={{ color: 'var(--ink-muted)' }}>
                Run context
              </p>
              <div className="mt-3 space-y-2 text-sm" style={{ color: 'var(--ink-base)' }}>
                <p>
                  <span style={{ color: 'var(--ink-muted)' }}>Uploaded:</span>{' '}
                  {new Date(job.created_at).toLocaleString()}
                </p>
                <p>
                  <span style={{ color: 'var(--ink-muted)' }}>Updated:</span>{' '}
                  {new Date(job.updated_at).toLocaleString()}
                </p>
                <p>
                  <span style={{ color: 'var(--ink-muted)' }}>Worker note:</span>{' '}
                  {progressNote ?? statusMeta.helper}
                </p>
              </div>
            </div>

            {isActive && (
              <div className="surface-card-muted p-5">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>Processing progress</p>
                  <span style={{ color: 'var(--ink-soft)' }}>Auto refresh</span>
                </div>
                <div className="mt-3 progress-track">
                  <div
                    className="progress-fill transition-all duration-700"
                    style={{
                      width: `${
                        job.status === 'created' ? 10
                        : job.status === 'uploaded' ? 20
                        : job.status === 'queued' ? 30
                        : 55
                      }%`,
                    }}
                  />
                </div>
                <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  {progressNote ?? statusMeta.helper}
                </p>
              </div>
            )}

            {job.error && (
              <div
                className="surface-card-muted p-5 text-sm"
                style={{ color: 'var(--danger)', background: 'var(--danger-dim)' }}
              >
                {job.error}
              </div>
            )}
          </aside>
        </div>
      </section>

      <section className="surface-card-strong p-3 flex flex-wrap gap-2">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className="rounded-full px-4 py-2 text-sm font-semibold transition-colors"
            style={{
              background: activeTab === tab.id ? 'rgba(255,255,255,0.1)' : 'transparent',
              color: activeTab === tab.id ? 'var(--ink-strong)' : 'var(--ink-soft)',
              border: activeTab === tab.id ? '1px solid var(--line-soft)' : '1px solid transparent',
            }}
          >
            {tab.label}
          </button>
        ))}
      </section>

      {activeTab === 'recap' && (
        <div className="grid gap-6 lg:grid-cols-[1.02fr_0.98fr]">
          <section className="surface-card p-6">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Run recap</p>
                <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                  What stands out first
                </h2>
              </div>
              {dashboard?.overview.smoothnessScore != null && (
                <span className="status-pill" style={{ color: 'var(--success)', background: 'var(--success-dim)' }}>
                  Smoothness {dashboard.overview.smoothnessScore}
                </span>
              )}
            </div>

            <p className="mt-4 text-base leading-7" style={{ color: 'var(--ink-base)' }}>
              {headline}
            </p>

            {dashboard ? (
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <div className="metric-tile">
                  <p className="metric-value">{dashboard.overview.bestTurnScore}</p>
                  <p className="metric-label">Best single-turn quality score</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{dashboard.overview.turnsDetected}</p>
                  <p className="metric-label">Turns included in the coaching pass</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{downloads.length}</p>
                  <p className="metric-label">Artifacts ready to inspect or export</p>
                </div>
                <div className="metric-tile">
                  <p className="metric-value">{coolMomentPhotos.length + peakFrames.length}</p>
                  <p className="metric-label">Key images surfaced from the run</p>
                </div>
              </div>
            ) : (
              <div className="mt-6 surface-card-muted p-4 text-sm" style={{ color: 'var(--ink-soft)' }}>
                Summary metrics will appear here once the summary artifact is available.
              </div>
            )}

            {!!dashboard?.warnings.length && (
              <div className="mt-6 surface-card-muted p-4">
                <p className="text-xs uppercase tracking-[0.22em]" style={{ color: 'var(--ink-muted)' }}>
                  Capture warnings
                </p>
                <ul className="mt-3 space-y-2 text-sm" style={{ color: 'var(--ink-base)' }}>
                  {dashboard.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          <section className="surface-card p-6">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Next focus</p>
                <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                  Practice cards
                </h2>
              </div>
              <span className="status-pill" style={{ color: 'var(--accent)', background: 'var(--accent-dim)' }}>
                Coaching tips
              </span>
            </div>

            <div className="mt-5 space-y-3">
              {(dashboard?.focusCards.length ? dashboard.focusCards : summary?.coaching_tips ?? []).slice(0, 4).map((tip) => {
                const category = tipCategory(tip)
                const catColors = CATEGORY_COLORS[category] ?? CATEGORY_COLORS.general
                const CATEGORY_LABELS: Record<string, string> = {
                  movement: 'Movement', edging: 'Edging', rhythm: 'Rhythm', balance: 'Balance', general: 'General',
                }
                return (
                  <div key={`${tip.title}-${tip.evidence}`} className={`coaching-card ${catColors.accent}`}>
                    <div className="flex items-center justify-between gap-3 pl-3">
                      <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>{tip.title}</p>
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 ${catColors.badge}`}>
                        {CATEGORY_LABELS[category]}
                      </span>
                    </div>
                    <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-base)' }}>
                      {tip.explanation}
                    </p>
                    <p className="mt-2 text-xs pl-3" style={{ color: 'var(--ink-muted)' }}>
                      {tip.evidence}
                    </p>
                    <Link
                      href="#"
                      className="mt-2 inline-flex items-center gap-1 text-xs pl-3"
                      style={{ color: 'var(--accent)' }}
                      onClick={(e) => { e.preventDefault(); setActiveTab('moments') }}
                    >
                      Watch
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                    </Link>
                  </div>
                )
              })}

              {!summary?.coaching_tips?.length && (
                <div className="surface-card-muted p-4 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  Tip cards appear when the summary JSON includes coaching guidance.
                </div>
              )}
            </div>
          </section>
        </div>
      )}

      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <section className="grid gap-4 lg:grid-cols-2">
            {dashboard?.categories.map((category) => (
              <article key={category.id} className="surface-card p-6">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-sm font-semibold" style={{ color: 'var(--ink-soft)' }}>{category.title}</p>
                    <p className="mt-2 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                      Current pipeline metrics mapped into a coaching-friendly bucket.
                    </p>
                  </div>
                  <div className="text-center">
                    <div
                      className="w-16 h-16 rounded-full flex items-center justify-center text-xl font-bold"
                      style={{ background: 'var(--accent-dim)', color: 'var(--ink-strong)' }}
                    >
                      {category.score}
                    </div>
                    <p className="mt-2 text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: 'var(--ink-muted)' }}>
                      {category.status}
                    </p>
                  </div>
                </div>

                <div className="mt-5 space-y-4">
                  {category.metrics.map((metric) => (
                    <div key={`${category.id}-${metric.label}`}>
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold" style={{ color: 'var(--snow-100)' }}>{metric.label}</p>
                        <p className="font-mono text-xs" style={{ color: 'var(--accent)' }}>{metric.value}</p>
                      </div>
                      <p className="mt-1 text-sm" style={{ color: 'var(--ink-soft)' }}>{metric.helper}</p>
                      <div className="mt-3 metric-rail">
                        <span style={{ width: `${metric.fill}%` }}>
                          <span className="metric-rail-dot" />
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-xs" style={{ color: 'var(--ink-muted)' }}>
                        <span>{metric.leftLabel}</span>
                        <span>{metric.rightLabel}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </article>
            )) ?? (
              <article className="surface-card p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
                Metrics will appear here once a summary artifact is attached.
              </article>
            )}
          </section>

          {!!dashboard?.turnHighlights.length && (
            <section className="surface-card p-6">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Turn highlights</p>
                  <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                    Best turns in this pass
                  </h2>
                </div>
                <span className="status-pill" style={{ color: 'var(--success)', background: 'var(--success-dim)' }}>
                  Technique scores
                </span>
              </div>

              <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                {dashboard.turnHighlights.map((turn) => (
                  <div key={turn.title} className="surface-card-muted p-4">
                    <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>{turn.title}</p>
                    <p className="mt-3 text-3xl font-bold tracking-tight" style={{ color: 'var(--ink-strong)', fontVariantNumeric: 'tabular-nums' }}>
                      {turn.score}
                    </p>
                    <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>{turn.detail}</p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}

      {activeTab === 'moments' && (
        <div className="space-y-6">
          <section className="surface-card p-6">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Key moments</p>
                <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                  Review the strongest still frames
                </h2>
              </div>
              <span className="status-pill" style={{ color: 'var(--accent)', background: 'var(--accent-dim)' }}>
                Motion gallery
              </span>
            </div>

            {coolMomentPhotos.length ? (
              <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {coolMomentPhotos.map((photo) => (
                  <a
                    key={photo.id}
                    href={photo.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="moment-card"
                  >
                    <img
                      src={photo.url}
                      alt={`Turn ${(photo.meta.turn_idx ?? 0) + 1}`}
                      className="w-full aspect-[4/3] object-cover"
                    />
                    <div className="moment-card-overlay">
                      <p className="text-xs font-mono text-white">
                        Turn {(photo.meta.turn_idx ?? 0) + 1}
                        {photo.meta.side ? ` · ${photo.meta.side}` : ''}
                        {photo.meta.timestamp_s != null ? ` · ${Number(photo.meta.timestamp_s).toFixed(1)}s` : ''}
                      </p>
                    </div>
                  </a>
                ))}
              </div>
            ) : (
              <div className="mt-5 surface-card-muted p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
                No cool-moment photos were attached to this run.
              </div>
            )}
          </section>

          <section className="surface-card p-6">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Peak pressure frames</p>
                <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                  Pressure snapshots across turns
                </h2>
              </div>
              <span className="status-pill" style={{ color: 'var(--gold)', background: 'var(--gold-dim)' }}>
                {peakFrames.length} frames
              </span>
            </div>

            {peakFrames.length ? (
              <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {peakFrames.map((frame) => (
                  <a
                    key={frame.id}
                    href={frame.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="moment-card"
                  >
                    <img
                      src={frame.url}
                      alt={`Turn ${(frame.meta.turn_idx ?? 0) + 1}`}
                      className="w-full aspect-[4/3] object-cover"
                    />
                    <div className="moment-card-overlay">
                      <p className="text-xs font-mono text-white">
                        Turn {(frame.meta.turn_idx ?? 0) + 1}
                        {frame.meta.side ? ` · ${frame.meta.side}` : ''}
                        {frame.meta.timestamp_s != null ? ` · ${Number(frame.meta.timestamp_s).toFixed(1)}s` : ''}
                      </p>
                    </div>
                  </a>
                ))}
              </div>
            ) : (
              <div className="mt-5 surface-card-muted p-6 text-sm" style={{ color: 'var(--ink-soft)' }}>
                Peak pressure frames have not been attached to this run yet.
              </div>
            )}
          </section>
        </div>
      )}

      {activeTab === 'downloads' && (
        <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <section className="surface-card p-6">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Exports</p>
              <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                Raw files from this run
              </h2>
            </div>

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
                    <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>{label}</p>
                    <p className="mt-1 text-xs" style={{ color: 'var(--ink-soft)' }}>
                      Open the signed artifact in a new tab.
                    </p>
                  </div>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--ink-soft)' }}>
                    <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
                  </svg>
                </a>
              )) : (
                <div className="surface-card-muted p-5 text-sm" style={{ color: 'var(--ink-soft)' }}>
                  No export files are ready yet.
                </div>
              )}
            </div>
          </section>

          <section className="surface-card p-6">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>Artifacts summary</p>
              <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--ink-strong)' }}>
                What this run produced
              </h2>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <div className="metric-tile">
                <p className="metric-value">{artifacts.length}</p>
                <p className="metric-label">Signed artifacts attached to the run.</p>
              </div>
              <div className="metric-tile">
                <p className="metric-value">{downloads.length}</p>
                <p className="metric-label">Immediate exports available from this page.</p>
              </div>
              <div className="metric-tile">
                <p className="metric-value">{coolMomentPhotos.length}</p>
                <p className="metric-label">Cool-moment photos surfaced by the pipeline.</p>
              </div>
              <div className="metric-tile">
                <p className="metric-value">{peakFrames.length}</p>
                <p className="metric-label">Peak pressure frames available for review.</p>
              </div>
            </div>

            {summary?.coaching_tips?.length ? (
              <div className="mt-6 surface-card-muted p-4">
                <p className="text-xs uppercase tracking-[0.22em]" style={{ color: 'var(--ink-muted)' }}>
                  Summary notes
                </p>
                <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                  {summary.coaching_tips[0].explanation}
                </p>
              </div>
            ) : null}
          </section>
        </div>
      )}
    </div>
  )
}
