'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { groupBySeason } from '@/lib/seasons'
import { scoreLabel } from '@/lib/analysis-summary'
import type { JobStatus } from '@/lib/types'
import { JobRetryAction } from './job-retry-action'
import { useLanguage } from '@/components/language-provider'
import { translateKnownText } from '@/lib/i18n'

export interface ArchiveRunItem {
  id: string
  created_at: string
  status: JobStatus
  statusLabel: string
  statusHelper: string
  statusTone: 'neutral' | 'accent' | 'warning' | 'success' | 'danger'
  statusDot: string
  statusPill: string
  canRetry: boolean
  actionLabel: string | null
  title?: string
  displayName?: string
  originalFilename?: string | null
  userNote?: string | null
  subtitle: string
  score: number | null
  scoreCountsForProgress: boolean
  previewUrl: string | null
  sessionType: string | null
  searchText?: string
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

function toneStyles(tone: ArchiveRunItem['statusTone']) {
  switch (tone) {
    case 'accent':
      return { color: 'var(--accent)', background: 'var(--accent-dim)', borderColor: 'rgba(0,132,212,0.15)' }
    case 'warning':
      return { color: 'var(--gold)', background: 'var(--gold-dim)', borderColor: 'rgba(199,154,68,0.22)' }
    case 'success':
      return { color: 'var(--success)', background: 'var(--success-dim)', borderColor: 'rgba(46,139,87,0.18)' }
    case 'danger':
      return { color: 'var(--danger)', background: 'var(--danger-dim)', borderColor: 'rgba(209,67,67,0.18)' }
    default:
      return { color: 'var(--ink-soft)', background: 'rgba(0,0,0,0.04)', borderColor: 'rgba(0,0,0,0.06)' }
  }
}

function fallbackIcon(statusDot: string) {
  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-[var(--radius-lg)]"
      style={{
        width: '4.75rem',
        height: '4.75rem',
        background: 'rgba(0,0,0,0.03)',
        border: '1px solid rgba(0,0,0,0.06)',
      }}
    >
      <div className="h-3 w-3 rounded-full" style={{ background: statusDot }} />
    </div>
  )
}

export function ArchiveRunsClient({
  runs,
  initialEditJobId = null,
}: {
  runs: ArchiveRunItem[]
  initialEditJobId?: string | null
}) {
  return <ArchiveRunsClientBody runs={runs} initialEditJobId={initialEditJobId} />
}

function ArchiveRunsClientBody({ runs, initialEditJobId }: { runs: ArchiveRunItem[]; initialEditJobId?: string | null }) {
  const { lang, dict } = useLanguage()
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | JobStatus>('all')
  const [sessionFilter, setSessionFilter] = useState('all')
  const [sortBy, setSortBy] = useState<'newest' | 'best'>('newest')

  const sessionOptions = useMemo(() => {
    const values = Array.from(new Set(runs.map((run) => run.sessionType).filter((value): value is string => Boolean(value))))
    return values.sort((left, right) => left.localeCompare(right))
  }, [runs])

  const filteredRuns = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase()
    const nextRuns = runs.filter((run) => {
      if (statusFilter !== 'all' && run.status !== statusFilter) return false
      if (sessionFilter !== 'all' && run.sessionType !== sessionFilter) return false
      if (!normalizedSearch) return true

      const searchHaystack = [
        run.displayName,
        run.title,
        run.originalFilename,
        run.userNote,
        run.subtitle,
        run.statusLabel,
        run.statusHelper,
        run.sessionType,
        run.searchText,
      ]
        .filter((value): value is string => Boolean(value))
        .join(' ')
        .toLowerCase()

      return searchHaystack.includes(normalizedSearch)
    })

    nextRuns.sort((left, right) => {
      if (sortBy === 'best') {
        const leftScore = left.score ?? -1
        const rightScore = right.score ?? -1
        if (rightScore !== leftScore) return rightScore - leftScore
      }
      return new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
    })

    return nextRuns
  }, [runs, search, sessionFilter, sortBy, statusFilter])

  const groups = groupBySeason(filteredRuns)

  return (
    <div className="space-y-6">
      <section className="surface-card p-5">
        <div className="grid gap-3 lg:grid-cols-[1.2fr_0.9fr_0.9fr_0.8fr]">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={dict.archive.search}
            className="select-input"
          />
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as 'all' | JobStatus)} className="select-input">
            <option value="all">{dict.archive.allStatuses}</option>
            <option value="done">{lang === 'zh' ? '已完成' : 'Completed'}</option>
            <option value="running">{lang === 'zh' ? '分析中' : 'Running'}</option>
            <option value="queued">{lang === 'zh' ? '排队中' : 'Queued'}</option>
            <option value="uploaded">{lang === 'zh' ? '已上传' : 'Uploaded'}</option>
            <option value="created">{lang === 'zh' ? '已创建' : 'Created'}</option>
            <option value="error">{lang === 'zh' ? '错误' : 'Error'}</option>
          </select>
          <select value={sessionFilter} onChange={(event) => setSessionFilter(event.target.value)} className="select-input">
            <option value="all">{dict.archive.allSessionTypes}</option>
            {sessionOptions.map((sessionType) => (
              <option key={sessionType} value={sessionType}>{sessionType}</option>
            ))}
          </select>
          <select value={sortBy} onChange={(event) => setSortBy(event.target.value as 'newest' | 'best')} className="select-input">
            <option value="newest">{dict.archive.newest}</option>
            <option value="best">{dict.archive.best}</option>
          </select>
        </div>
      </section>

      {!filteredRuns.length ? (
        <section className="surface-card p-6">
          <div className="surface-card-muted p-10 text-center">
            <p className="text-base font-bold" style={{ color: 'var(--ink-strong)' }}>{dict.archive.noMatchTitle}</p>
            <p className="text-sm mt-2" style={{ color: 'var(--ink-soft)' }}>
              {dict.archive.noMatchBody}
            </p>
          </div>
        </section>
      ) : (
        groups.map((group) => {
          const groupRuns = group.runs as ArchiveRunItem[]
          const groupScored = groupRuns.filter((run): run is ArchiveRunItem & { score: number } => run.score != null && run.scoreCountsForProgress)
          const groupAvg = groupScored.length
            ? Math.round(groupScored.reduce((sum, run) => sum + run.score, 0) / groupScored.length)
            : null
          const groupBest = groupScored.length ? Math.max(...groupScored.map((run) => run.score)) : null

          return (
            <section key={group.label} className="surface-card p-6">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3">
                  <div>
                    <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--ink-muted)' }}>
                      {groupRuns.length} {groupRuns.length === 1 ? dict.archive.run : dict.archive.runs}
                    </p>
                    <h2 className="mt-1" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {group.label}
                    </h2>
                  </div>
                  {groupAvg != null && (
                    <span className={levelBadgeClass(scoreLabel(groupAvg))} style={{ marginLeft: '0.5rem' }}>
                      {translateKnownText(scoreLabel(groupAvg), lang)}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  {groupAvg != null && (
                    <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                      {dict.archive.avg} <span className="font-bold" style={{ color: 'var(--accent)' }}>{groupAvg}</span>
                    </span>
                  )}
                  {groupBest != null && (
                    <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>
                      {dict.archive.bestLabel} <span className="font-bold" style={{ color: 'var(--success)' }}>{groupBest}</span>
                    </span>
                  )}
                </div>
              </div>

              <ul className="space-y-3 mt-5">
                {groupRuns.map((run) => {
                  const statusStyle = toneStyles(run.statusTone)
                  const displayName = run.displayName ?? run.title ?? dict.archive.untitled
                  const scoreExcluded = run.score != null && !run.scoreCountsForProgress
                  return (
                    <li key={run.id}>
                      <div
                        className="surface-card-muted flex items-center gap-4 px-4 py-4 group hover:-translate-y-0.5"
                        style={{
                          display: 'flex',
                          transition: 'transform 150ms ease, background 0.15s ease, border-color 0.15s ease',
                          boxShadow: initialEditJobId === run.id ? '0 0 0 1px rgba(0,132,212,0.18) inset' : undefined,
                        }}
                      >
                        <Link href={`/jobs/${run.id}`} className="flex min-w-0 flex-1 items-center gap-4">
                          {run.previewUrl ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                              src={run.previewUrl}
                              alt={displayName}
                              className="shrink-0 rounded-[var(--radius-lg)] object-cover"
                              style={{ width: '4.75rem', height: '4.75rem' }}
                            />
                          ) : fallbackIcon(run.statusDot)}

                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <p className="text-sm font-semibold truncate" style={{ color: 'var(--ink-strong)' }}>{displayName}</p>
                              {run.sessionType && (
                                <span
                                  className="text-[0.68rem] font-semibold px-2 py-0.5 rounded-full"
                                  style={{ background: 'rgba(0,0,0,0.05)', color: 'var(--ink-soft)' }}
                                >
                                  {run.sessionType}
                                </span>
                              )}
                            </div>
                            {run.originalFilename && run.originalFilename !== displayName && (
                              <p className="mt-1 text-xs truncate" style={{ color: 'var(--ink-muted)' }}>
                                {run.originalFilename}
                              </p>
                            )}
                            <p className="text-xs mt-1" style={{ color: 'var(--ink-soft)' }}>
                              {run.subtitle}
                            </p>
                            <p className="mt-1 text-xs leading-5" style={{ color: 'var(--ink-base)' }}>
                              {translateKnownText(run.statusHelper, lang)}
                            </p>
                          </div>
                        </Link>

                        <div className="flex shrink-0 flex-col items-end gap-2">
                          {run.score != null && (
                            <div className="text-right">
                              <p className="text-base font-bold" style={{ color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
                                {run.score}
                              </p>
                              <p className="text-xs" style={{ color: 'var(--ink-muted)' }}>
                                {scoreExcluded
                                  ? dict.archive.excludedFromProgress
                                  : translateKnownText(scoreLabel(run.score), lang)}
                              </p>
                            </div>
                          )}

                          <span
                            className="shrink-0 rounded-full border px-2.5 py-1 text-xs font-semibold"
                            style={{ background: statusStyle.background, color: statusStyle.color, borderColor: statusStyle.borderColor }}
                          >
                            {translateKnownText(run.statusLabel, lang)}
                          </span>

                          <JobRetryAction
                            jobId={run.id}
                            canRetry={run.canRetry}
                            actionLabel={run.actionLabel}
                            compact
                          />
                        </div>
                      </div>
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
