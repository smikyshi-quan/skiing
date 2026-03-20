'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { LogoutButton } from './logout-button'

export function NavLinks() {
  const pathname = usePathname()

  return (
    <nav className="topnav">
      <Link
        href="/upload"
        className={`topnav-link ${pathname === '/upload' ? 'topnav-link--active' : ''}`}
      >
        Analyse
      </Link>
      <Link
        href="/jobs"
        className={`topnav-link ${pathname.startsWith('/jobs') ? 'topnav-link--active' : ''}`}
      >
        Archive
      </Link>
      <LogoutButton />
    </nav>
  )
}
