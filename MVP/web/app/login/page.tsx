'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

type Mode = 'login' | 'signup'

export default function LoginPage() {
  const router = useRouter()

  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState<{ text: string; kind: 'error' | 'info' } | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setMessage(null)
    setLoading(true)
    try {
      const supabase = createClient()
      if (mode === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
        router.push('/')
        router.refresh()
      } else {
        const { error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        setMessage({ text: 'Check your email for a confirmation link.', kind: 'info' })
      }
    } catch (err: unknown) {
      setMessage({
        text: err instanceof Error ? err.message : 'An error occurred',
        kind: 'error',
      })
    } finally {
      setLoading(false)
    }
  }

  const previewTags = [
    'Edge angle',
    'Turn rhythm',
    'Upper-body quietness',
    'Balance',
    'Pressure timing',
    'Progress tracking',
  ]

  return (
    <>
      <div className="route-bg route-bg--login" />
      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        {/* Marketing panel */}
        <section className="surface-card p-8 lg:p-10 overflow-hidden">
          <span className="eyebrow eyebrow--warm">AI ski coach in your pocket</span>
          <div className="mt-6 max-w-xl">
            <h1 className="section-title">
              Bring every run back<br />with clearer feedback.
            </h1>
            <p className="section-copy mt-4">
              Upload a clip, review your movement, and turn each session into a sharper practice plan.
            </p>
          </div>

          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/alpine/hero-corduroy.jpg"
            alt="Groomed corduroy slope"
            className="hero-photo mt-6"
            style={{ height: '180px' }}
          />

          <div className="mt-6 flex flex-wrap gap-2">
            {previewTags.map((tag) => (
              <span key={tag} className="accent-chip">{tag}</span>
            ))}
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <div className="metric-tile">
              <p className="metric-value">17+</p>
              <p className="metric-label">Technique markers surfaced across a run recap</p>
            </div>
            <div className="metric-tile">
              <p className="metric-value">3</p>
              <p className="metric-label">Core outputs: recap, key moments, and progress tracking</p>
            </div>
            <div className="metric-tile">
              <p className="metric-value">1</p>
              <p className="metric-label">Single upload to start the whole coaching loop</p>
            </div>
          </div>
        </section>

        {/* Auth panel */}
        <section className="surface-card-strong p-6 lg:p-8 self-start">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold" style={{ color: 'var(--ink-soft)' }}>
                {mode === 'login' ? 'Welcome back' : 'Create your account'}
              </p>
              <h2 className="mt-1 text-2xl font-extrabold tracking-tight" style={{ color: 'var(--ink-strong)' }}>
                {mode === 'login' ? 'Review your latest run.' : 'Start your first analysis.'}
              </h2>
            </div>
            <span className="status-pill" style={{ color: 'var(--amber)', background: 'var(--amber-dim)' }}>
              {mode === 'login' ? 'Return' : 'New athlete'}
            </span>
          </div>

          <p className="mt-3 text-sm" style={{ color: 'var(--ink-soft)' }}>
            {mode === 'login'
              ? 'Sign in to open your coaching hub, latest recap, and saved runs.'
              : 'Create an account so your uploads, results, and coaching cards stay synced.'}
          </p>

          <form onSubmit={handleSubmit} className="mt-7 space-y-4">
            <div>
              <label className="field-label">Email</label>
              <input
                type="email"
                required
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="text-input"
              />
            </div>

            <div>
              <label className="field-label">Password</label>
              <input
                type="password"
                required
                minLength={6}
                placeholder="••••••••"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="text-input"
              />
            </div>

            {message && (
              <div
                className="rounded-2xl px-4 py-3 text-sm"
                style={{
                  background: message.kind === 'error' ? 'var(--danger-dim)' : 'var(--success-dim)',
                  color: message.kind === 'error' ? 'var(--danger)' : 'var(--success)',
                  border: `1px solid ${message.kind === 'error' ? 'rgba(209,67,67,0.2)' : 'rgba(46,139,87,0.2)'}`,
                }}
              >
                {message.text}
              </div>
            )}

            <button type="submit" disabled={loading} className="cta-primary w-full">
              {loading ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>

          <div
            className="mt-6 rounded-[var(--radius-lg)] px-4 py-4"
            style={{ background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.06)' }}
          >
            <p className="text-xs font-bold uppercase tracking-[0.22em]" style={{ color: 'var(--ink-muted)' }}>
              The coaching loop
            </p>
            <div className="mt-3 space-y-3 text-sm" style={{ color: 'var(--ink-base)' }}>
              <div className="flex items-center justify-between">
                <span>Upload your run</span>
                <span style={{ color: 'var(--ink-muted)' }}>01</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Review technique recap</span>
                <span style={{ color: 'var(--ink-muted)' }}>02</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Focus on what matters next</span>
                <span style={{ color: 'var(--ink-muted)' }}>03</span>
              </div>
            </div>
          </div>

          <p className="mt-5 text-sm" style={{ color: 'var(--ink-soft)' }}>
            {mode === 'login' ? (
              <>
                No account yet?{' '}
                <button
                  onClick={() => { setMode('signup'); setMessage(null) }}
                  className="font-semibold hover:underline"
                  style={{ color: 'var(--accent)' }}
                >
                  Create one
                </button>
              </>
            ) : (
              <>
                Already tracking runs?{' '}
                <button
                  onClick={() => { setMode('login'); setMessage(null) }}
                  className="font-semibold hover:underline"
                  style={{ color: 'var(--accent)' }}
                >
                  Sign in
                </button>
              </>
            )}
          </p>
        </section>
      </div>
    </>
  )
}
