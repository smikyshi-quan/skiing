import type { Metadata } from 'next'
import Link from 'next/link'
import { LogoutButton } from '@/components/logout-button'
import './globals.css'

export const metadata: Metadata = {
  title: 'SkiCoach AI',
  description: 'Video-based ski coaching with alpine run recaps, moments, and progress tracking.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="site-shell">
          <header className="site-topbar">
            <Link href="/" className="brand-lockup">
              <span className="brand-mark">S</span>
              <span className="brand-wordmark">
                <strong>SkiCoach AI</strong>
                <span>Alpine Coach</span>
              </span>
            </Link>

            <nav className="topnav">
              <Link href="/upload" className="topnav-link">Analyse</Link>
              <Link href="/jobs" className="topnav-link">Archive</Link>
              <LogoutButton />
            </nav>
          </header>

          <main className="page-shell">{children}</main>
        </div>
      </body>
    </html>
  )
}
