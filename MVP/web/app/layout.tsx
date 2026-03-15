import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Ski Technique Analyser',
  description: 'AI-powered ski coaching — peak pressure frame analysis',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900 antialiased">
        <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-6">
          <a href="/jobs" className="font-semibold text-base tracking-tight">
            SkiCoach
          </a>
          <nav className="flex gap-4 text-sm text-gray-500 ml-auto">
            <a href="/upload" className="hover:text-gray-900 transition-colors">
              Upload
            </a>
            <a href="/jobs" className="hover:text-gray-900 transition-colors">
              Jobs
            </a>
          </nav>
        </header>
        <main className="max-w-4xl mx-auto px-4 py-10">{children}</main>
      </body>
    </html>
  )
}
