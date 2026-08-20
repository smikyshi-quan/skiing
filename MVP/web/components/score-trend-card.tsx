import type { Job } from '@/lib/types'
import type { Lang } from '@/lib/i18n'
import { getDictionary } from '@/lib/i18n'

function average(values: number[]) {
  if (!values.length) return null
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length)
}

function linePoints(values: number[]) {
  if (!values.length) return ''

  const width = 320
  const height = 124
  const paddingX = 12
  const paddingY = 14
  const min = Math.min(...values, 0)
  const max = Math.max(...values, 100)
  const range = Math.max(max - min, 1)

  return values.map((value, index) => {
    const x = paddingX + (index * (width - paddingX * 2)) / Math.max(values.length - 1, 1)
    const y = height - paddingY - ((value - min) / range) * (height - paddingY * 2)
    return `${x},${y}`
  }).join(' ')
}

export function ScoreTrendCard({
  runs,
  title = 'Progress over time',
  subtitle = 'Your last 10 scored runs.',
  lang = 'en',
}: {
  runs: Array<Pick<Job, 'score' | 'created_at'> & {
    scoreCountsForProgress?: boolean
    score_counts_for_progress?: boolean
  }>
  title?: string
  subtitle?: string
  lang?: Lang
}) {
  const dict = getDictionary(lang)
  const trendRuns = runs
    .filter((run): run is Pick<Job, 'score' | 'created_at'> & { score: number } => (
      run.score != null
      && run.scoreCountsForProgress !== false
      && run.score_counts_for_progress !== false
    ))
    .slice(0, 10)
    .reverse()

  if (trendRuns.length < 2) {
    return (
      <section className="surface-card p-6">
        <p className="section-label">{title}</p>
        <h2 className="mt-2" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
          {dict.trend.buildTitle}
        </h2>
        <p className="mt-3 text-sm leading-6" style={{ color: 'var(--ink-soft)' }}>
          {dict.trend.buildBody}
        </p>
      </section>
    )
  }

  const values = trendRuns.map((run) => run.score)
  const latest = values[values.length - 1] ?? null
  const best = Math.max(...values)
  const recentAverage = average(values.slice(-5))
  const previousAverage = average(values.slice(Math.max(0, values.length - 10), Math.max(0, values.length - 5)))
  const delta = recentAverage != null && previousAverage != null ? recentAverage - previousAverage : null

  return (
    <section className="surface-card p-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="section-label">{title}</p>
          <h2 className="mt-2" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
            {latest != null ? `${dict.trend.latestScore} ${latest}` : dict.trend.yourTrend}
          </h2>
          <p className="mt-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
            {subtitle}
          </p>
        </div>
        {delta != null && (
          <span
            className="text-xs font-bold px-2.5 py-1 rounded-full"
            style={{
              color: delta >= 0 ? 'var(--success)' : 'var(--danger)',
              background: delta >= 0 ? 'var(--success-dim)' : 'var(--danger-dim)',
            }}
          >
            {delta >= 0 ? '+' : ''}{delta} {dict.trend.delta}
          </span>
        )}
      </div>

      <div className="mt-5 overflow-hidden rounded-[var(--radius-lg)] border" style={{ borderColor: 'rgba(17,17,17,0.08)', background: 'rgba(255,255,255,0.9)' }}>
        <svg viewBox="0 0 320 124" className="w-full h-32">
          <line x1="12" y1="110" x2="308" y2="110" stroke="rgba(17,17,17,0.08)" strokeWidth="1" />
          <line x1="12" y1="62" x2="308" y2="62" stroke="rgba(17,17,17,0.05)" strokeWidth="1" strokeDasharray="4 4" />
          <polyline
            fill="none"
            stroke="var(--accent)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={linePoints(values)}
          />
          {linePoints(values).split(' ').map((point, index) => {
            if (!point) return null
            const [cx, cy] = point.split(',')
            return (
              <circle
                key={`${point}-${index}`}
                cx={cx}
                cy={cy}
                r="4"
                fill="#fff"
                stroke="var(--accent)"
                strokeWidth="2"
              />
            )
          })}
        </svg>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="metric-tile">
          <p className="metric-value" style={{ fontSize: '1.7rem' }}>{best}</p>
          <p className="metric-label">{dict.trend.bestWindow}</p>
        </div>
        <div className="metric-tile">
          <p className="metric-value" style={{ fontSize: '1.7rem' }}>{recentAverage ?? '—'}</p>
          <p className="metric-label">{dict.trend.averageLastFive}</p>
        </div>
        <div className="metric-tile">
          <p className="metric-value" style={{ fontSize: '1.7rem' }}>{trendRuns.length}</p>
          <p className="metric-label">{dict.trend.runsShown}</p>
        </div>
      </div>
    </section>
  )
}
