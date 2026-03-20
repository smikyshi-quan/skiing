import type { Metadata } from 'next'
import Link from 'next/link'
import { NavLinks } from '@/components/nav-links'
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

            <NavLinks />
          </header>

          <main className="page-shell">{children}</main>
        </div>
      </body>
    </html>
  )
}
