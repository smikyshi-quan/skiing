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
        router.push('/jobs')
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

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Logo mark */}
        <div className="flex justify-center mb-8">
          <div
            className="w-12 h-12 rounded-2xl flex items-center justify-center font-bold text-xl text-white"
            style={{ background: 'var(--accent)' }}
          >
            S
          </div>
        </div>

        <h1 className="text-2xl font-bold text-center text-white mb-1">
          {mode === 'login' ? 'Welcome back' : 'Create your account'}
        </h1>
        <p className="text-sm text-center mb-8" style={{ color: 'rgba(255,255,255,0.4)' }}>
          {mode === 'login'
            ? 'Sign in to review your ski analysis'
            : 'Start analysing your ski technique'}
        </p>

        <div className="card p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'rgba(255,255,255,0.4)' }}>
                Email
              </label>
              <input
                type="email"
                required
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-dark"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'rgba(255,255,255,0.4)' }}>
                Password
              </label>
              <input
                type="password"
                required
                minLength={6}
                placeholder="••••••••"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-dark"
              />
            </div>

            {message && (
              <div
                className="text-sm rounded-xl px-4 py-3"
                style={{
                  background: message.kind === 'error'
                    ? 'rgba(239,68,68,0.12)'
                    : 'rgba(34,208,122,0.12)',
                  color: message.kind === 'error' ? '#F87171' : '#4ADE80',
                  border: `1px solid ${message.kind === 'error' ? 'rgba(239,68,68,0.3)' : 'rgba(34,208,122,0.3)'}`,
                }}
              >
                {message.text}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full mt-2"
              style={{ padding: '0.75rem 1.25rem' }}
            >
              {loading ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>
        </div>

        <p className="text-sm text-center mt-5" style={{ color: 'rgba(255,255,255,0.35)' }}>
          {mode === 'login' ? (
            <>
              No account?{' '}
              <button
                onClick={() => { setMode('signup'); setMessage(null) }}
                className="font-medium hover:underline"
                style={{ color: 'var(--accent)' }}
              >
                Sign up
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button
                onClick={() => { setMode('login'); setMessage(null) }}
                className="font-medium hover:underline"
                style={{ color: 'var(--accent)' }}
              >
                Sign in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  )
}
