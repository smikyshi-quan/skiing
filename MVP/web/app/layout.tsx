import type { Metadata } from 'next'
import { Manrope } from 'next/font/google'
import Link from 'next/link'
import { NavLinks } from '@/components/nav-links'
import './globals.css'

const manrope = Manrope({
  subsets: ['latin'],
  variable: '--font-manrope',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'SkiCoach AI',
  description: 'Video-based ski coaching with alpine run recaps, moments, and progress tracking.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={manrope.variable}>
      <body>
        <div className="site-shell">
          <header className="site-topbar">
            <Link href="/" className="brand-lockup">
              <span className="brand-mark">S</span>
              <span className="brand-wordmark">
                <strong>SkiCoach AI</strong>
              </span>
            </Link>
            <span className="brand-sep" />
            <span style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase' as const, color: 'var(--ink-muted)' }}>
              Alpine Coach
            </span>

            <NavLinks />
          </header>

          <main className="page-shell">{children}</main>
        </div>
      </body>
    </html>
  )
}
