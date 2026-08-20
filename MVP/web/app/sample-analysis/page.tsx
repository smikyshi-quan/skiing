'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  buildTechniqueDashboard,
  clipQualityLabel,
  scoreLabel,
  type AiCoachingPoint,
} from '@/lib/analysis-summary'
import { SiteFooter } from '@/components/site-footer'
import { useLanguage } from '@/components/language-provider'
import { scoreContextForLang, translateKnownText } from '@/lib/i18n'
import { getDrill, localizeDrill } from '@/lib/drills'
import { getSampleCoaching, SAMPLE_OVERLAY_PATH, SAMPLE_SUMMARY } from '@/lib/sample-run'

const dashboard = buildTechniqueDashboard(SAMPLE_SUMMARY)
const score = dashboard.overview.overallScore
const level = scoreLabel(score)

const CATEGORY_COLORS: Record<string, { accent: string; badge: string }> = {
  balance: { accent: 'coaching-accent-balance', badge: 'category-badge-balance' },
  edging: { accent: 'coaching-accent-edging', badge: 'category-badge-edging' },
  rhythm: { accent: 'coaching-accent-rhythm', badge: 'category-badge-rhythm' },
  movement: { accent: 'coaching-accent-movement', badge: 'category-badge-movement' },
  general: { accent: 'coaching-accent-general', badge: 'category-badge-general' },
}

const CATEGORY_LABELS: Record<string, string> = {
  movement: 'Movement', edging: 'Edging', rhythm: 'Rhythm', balance: 'Balance', general: 'General',
}

function metricDotColor(value: number, threshold: number): string {
  return value >= threshold ? 'var(--accent)' : 'var(--gold)'
}

type Tab = 'recap' | 'metrics'

export default function SampleAnalysisPage() {
  const { lang, dict } = useLanguage()
  const [activeTab, setActiveTab] = useState<Tab>('recap')
  const [showAllTips, setShowAllTips] = useState(false)
  const sampleCoaching = getSampleCoaching(lang)

  // Headline mirrors the recap page: the qualitative level, not the judge's
  // top coaching point (that already leads the coaching section below).
  const headline = translateKnownText(level, lang)

  return (
    <>
      <div className="route-bg route-bg--detail" />
      <div className="space-y-6">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 px-1 sm:px-2 text-sm" style={{ color: 'var(--ink-soft)' }}>
          <Link href="/" className="hover:underline">{dict.sample.home}</Link>
          <span>/</span>
          <span style={{ color: 'var(--ink-strong)' }}>{dict.sample.page}</span>
        </div>

        {/* ── Hero: video + score column ──────────────── */}
        <section className="surface-card p-6 lg:p-7">
          <div className="grid gap-6 lg:grid-cols-[1.16fr_0.84fr]">
            {/* Left: video + score */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span className="eyebrow">{dict.sample.runRecap}</span>
                <span className="eyebrow eyebrow--warm" style={{ fontSize: '0.62rem', padding: '0.3rem 0.6rem' }}>
                  {dict.sample.sampleRun}
                </span>
              </div>

              {/* Score + headline row */}
              <div className="flex items-start gap-5">
                <div className="score-ring shrink-0" style={{ width: '6.5rem', height: '6.5rem' }}>
                  <div className="score-ring-glow" />
                  <svg width="104" height="104" viewBox="0 0 104 104">
                    <circle cx="52" cy="52" r="44" fill="none" stroke="rgba(0,0,0,0.06)" strokeWidth="6" />
                    <circle
                      cx="52" cy="52" r="44"
                      fill="none"
                      stroke="url(#scoreGradSample)"
                      strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray="276.46"
                      strokeDashoffset={276.46 - (score / 100) * 276.46}
                    />
                    <defs>
                      <linearGradient id="scoreGradSample" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="#0084d4" />
                        <stop offset="100%" stopColor="#c79a44" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="score-ring-label">
                    <span className="font-extrabold tracking-tight" style={{ fontSize: '1.6rem', color: 'var(--ink-strong)' }}>
                      {score}
                    </span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h1 style={{ fontSize: 'clamp(1.3rem, 2.4vw, 1.8rem)', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.2, color: 'var(--ink-strong)' }}>
                    {headline}
                  </h1>
                  <div className="mt-2 flex items-center gap-2 flex-wrap">
                    <span className="text-xs" style={{ color: 'var(--ink-soft)' }}>
                      {scoreContextForLang(score, lang)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Overlay video */}
              <div
                className="overflow-hidden"
                style={{ borderRadius: 'var(--radius-xl)', background: '#0a0f1a', border: '1px solid rgba(255,255,255,0.15)' }}
              >
                <video src={SAMPLE_OVERLAY_PATH} autoPlay loop muted playsInline controls className="w-full aspect-video bg-black" />
              </div>
            </div>

            {/* Right sidebar */}
            <aside className="space-y-4">
              {/* Quick metrics */}
              <div className="surface-card-muted p-5">
                <p className="section-label">{dict.sample.summary}</p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className={dashboard.overview.overallScore > 60 ? 'metric-tile metric-tile--high' : 'metric-tile metric-tile--low'}>
                    <div className="metric-tile-dot" style={{ background: metricDotColor(dashboard.overview.overallScore, 60) }} />
                    <p className="metric-value" style={{ color: dashboard.overview.overallScore > 60 ? 'var(--accent)' : 'var(--gold)' }}>
                      {dashboard.overview.overallScore}
                    </p>
                    <p className="metric-label">{dict.sample.techniqueScore}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard.overview.turnsDetected}</p>
                    <p className="metric-label">{dict.sample.turnsDetected}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard.overview.edgeAngle.toFixed(0)}&deg;</p>
                    <p className="metric-label">{dict.sample.edgeAngle}</p>
                  </div>
                  <div className="metric-tile metric-tile--high">
                    <div className="metric-tile-dot" style={{ background: 'var(--accent)' }} />
                    <p className="metric-value" style={{ color: 'var(--accent)' }}>
                      {translateKnownText(clipQualityLabel(dashboard.reliability), lang)}
                    </p>
                    <p className="metric-label">{dict.sample.clipQuality}</p>
                  </div>
                </div>
              </div>

              {/* Run context (sample-specific) */}
              <div className="surface-card-muted p-5">
                <p className="section-label">{dict.sample.about}</p>
                <div className="mt-3 space-y-2 text-sm" style={{ color: 'var(--ink-base)' }}>
                  <p>{dict.sample.aboutBody1} {dict.sample.aboutBody2}</p>
                  <p style={{ color: 'var(--ink-muted)' }}>
                    {dict.sample.aboutBody3}
                  </p>
                </div>
              </div>
            </aside>
          </div>
        </section>

        {/* ── Tab navigation ──────────────────────────── */}
        <section className="surface-card-strong p-3 flex flex-wrap gap-2" style={{ position: 'sticky', top: 'var(--sticky-tabs-offset)', zIndex: 30 }}>
          {([{ id: 'recap' as Tab, label: dict.sample.recap }, { id: 'metrics' as Tab, label: dict.sample.metrics }]).map((tab) => (
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
              {tab.label}
            </button>
          ))}
        </section>

        {/* ── Recap tab ───────────────────────────────── */}
        {activeTab === 'recap' && (
          <div className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-[1.02fr_0.98fr]">
              {/* Run summary */}
              <section className="surface-card p-6">
                <p className="section-label">{dict.sample.runSummary}</p>
                <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                  {dict.sample.whatStandsOut}
                </h2>
                <p className="mt-4 text-base leading-7" style={{ color: 'var(--ink-base)' }}>
                  {sampleCoaching.coach_summary}
                </p>
                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard.overview.bestTurnScore}</p>
                    <p className="metric-label">{dict.sample.bestTurn}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard.overview.turnsDetected}</p>
                    <p className="metric-label">{dict.sample.turnsInPass}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{sampleCoaching.coaching_points.length}</p>
                    <p className="metric-label">{dict.sample.priorities}</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-value">{dashboard.overview.smoothnessScore ?? '—'}</p>
                    <p className="metric-label">{dict.sample.turnFlow}</p>
                  </div>
                </div>
              </section>

              {/* Coaching insights */}
              <section className="surface-card p-6">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label" style={{ color: 'var(--amber)' }}>{dict.sample.coachingInsights}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.sample.priorities}
                    </h2>
                  </div>
                </div>

                <div className="mt-5 space-y-3">
                  {(() => {
                    const allTips = sampleCoaching.coaching_points
                    const visibleTips = showAllTips ? allTips : allTips.slice(0, 2)
                    return (
                      <>
                        {visibleTips.map((tip: AiCoachingPoint, idx) => {
                          const category = tip.category ?? 'general'
                          const catColors = CATEGORY_COLORS[category] ?? CATEGORY_COLORS.general
                          const drill = tip.recommended_drill_id ? getDrill(tip.recommended_drill_id) : null
                          const localizedDrill = drill ? localizeDrill(drill, lang) : null
                          return (
                            <div key={`${tip.title}-${idx}`} className={`coaching-card ${catColors.accent}`}>
                              <div className="flex items-center gap-3 pl-3">
                                <span className="preflight-number" style={{ width: '1.5rem', height: '1.5rem', fontSize: '0.62rem' }}>
                                  {String(idx + 1).padStart(2, '0')}
                                </span>
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>{tip.title}</p>
                                </div>
                                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 ${catColors.badge}`}>
                                  {translateKnownText(CATEGORY_LABELS[category], lang)}
                                </span>
                              </div>
                              <p className="mt-2 text-sm leading-6 pl-3" style={{ color: 'var(--ink-base)' }}>
                                {tip.feedback}
                              </p>
                              {localizedDrill && (
                                <div className="mt-3 pl-3">
                                  <div className="drill-card inline-flex items-center gap-3 px-4 py-3">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
                                        {dict.job.recommendedDrill}
                                      </p>
                                      <p className="mt-1 text-sm font-bold" style={{ color: 'var(--ink-strong)' }}>
                                        {localizedDrill.title}
                                      </p>
                                      <p className="mt-1 text-xs" style={{ color: 'var(--ink-soft)' }}>
                                        {localizedDrill.description}
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          )
                        })}
                        {allTips.length > 2 && !showAllTips && (
                          <button
                            type="button"
                            onClick={() => setShowAllTips(true)}
                            className="text-sm font-semibold px-3 py-2 rounded-xl transition-colors"
                            style={{ color: 'var(--accent)', background: 'var(--accent-dim)' }}
                          >
                            {dict.sample.showAll.replace('{count}', String(allTips.length))}
                          </button>
                        )}
                        {showAllTips && allTips.length > 2 && (
                          <button
                            type="button"
                            onClick={() => setShowAllTips(false)}
                            className="text-sm font-semibold px-3 py-2 rounded-xl transition-colors"
                            style={{ color: 'var(--ink-soft)', background: 'rgba(0,0,0,0.04)' }}
                          >
                            {dict.sample.showTopTwo}
                          </button>
                        )}
                      </>
                    )
                  })()}
                </div>

                {sampleCoaching.additional_observations?.length > 0 && (
                  <div className="mt-6">
                    <p className="section-label">{dict.job.additional}</p>
                    <ul className="mt-3 space-y-2">
                      {sampleCoaching.additional_observations.map((observation, idx) => (
                        <li
                          key={`${observation}-${idx}`}
                          className="text-sm leading-6 pl-3"
                          style={{ color: 'var(--ink-base)', borderLeft: '2px solid rgba(0,0,0,0.08)', paddingLeft: '0.75rem' }}
                        >
                          {observation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </div>
          </div>
        )}

        {/* ── Metrics tab ───────────────────────────────── */}
        {activeTab === 'metrics' && (
          <div className="space-y-6">
            <section className="grid gap-4 lg:grid-cols-2">
              {dashboard.categories.map((category) => (
                <article key={category.id} className="surface-card p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="section-label">{translateKnownText(category.title, lang)}</p>
                      <p className="mt-2 text-sm leading-6" style={{ color: 'var(--ink-base)' }}>
                        {dict.sample.checksBody}
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
                          <p className="font-mono text-xs" style={{ color: 'var(--accent)' }}>{translateKnownText(metric.value, lang)}</p>
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
              ))}
            </section>

            {!!dashboard.turnHighlights.length && (
              <section className="surface-card p-6">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="section-label">{dict.sample.turnHighlights}</p>
                    <h2 className="mt-2" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-strong)' }}>
                      {dict.sample.bestTurns}
                    </h2>
                  </div>
                  <span className="status-pill" style={{ color: 'var(--success)', background: 'var(--success-dim)' }}>
                    {dict.sample.techniqueScores}
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

        {/* CTA bar */}
        <section className="surface-card-strong p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold" style={{ color: 'var(--ink-strong)' }}>
              {dict.sample.ctaTitle}
            </p>
            <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>
              {dict.sample.ctaBody}
            </p>
          </div>
          <div className="flex gap-3">
            <Link href="/login" className="cta-secondary" style={{ padding: '0.6rem 1.2rem', fontSize: '0.85rem' }}>
              {dict.sample.viewPricing}
            </Link>
            <Link href="/login" className="cta-primary" style={{ padding: '0.6rem 1.2rem', fontSize: '0.85rem' }}>
              {dict.sample.signUp}
            </Link>
          </div>
        </section>

        {/* Footer */}
        <SiteFooter lang={lang} />
      </div>
    </>
  )
}
