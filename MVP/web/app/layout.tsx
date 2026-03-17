import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SkiCoach AI',
  description: 'AI-powered ski coaching — analyse your run',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased" style={{ backgroundColor: 'var(--bg-base)', color: '#fff' }}>
        <header
          className="flex items-center gap-6 px-6 py-4 sticky top-0 z-50"
          style={{
            backgroundColor: 'rgba(7,9,15,0.85)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            borderBottom: '1px solid var(--border-subtle)',
          }}
        >
          {/* Logo */}
          <a href="/jobs" className="flex items-center gap-2 group">
            <span
              className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm"
              style={{ background: 'var(--accent)' }}
            >
              S
            </span>
            <span className="font-bold text-sm tracking-tight text-white">
              SkiCoach <span style={{ color: 'var(--accent)' }}>AI</span>
            </span>
          </a>

          <nav className="flex items-center gap-1 ml-auto">
            <a
              href="/upload"
              className="text-sm px-3 py-1.5 rounded-lg transition-colors"
              style={{ color: 'rgba(255,255,255,0.55)' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.55)')}
            >
              Upload
            </a>
            <a
              href="/jobs"
              className="text-sm px-3 py-1.5 rounded-lg transition-colors"
              style={{ color: 'rgba(255,255,255,0.55)' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.55)')}
            >
              My Runs
            </a>
          </nav>
        </header>

        <main className="max-w-3xl mx-auto px-4 py-10">{children}</main>
      </body>
    </html>
  )
}
