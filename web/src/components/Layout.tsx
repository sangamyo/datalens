import { NavLink, Outlet } from 'react-router-dom'
import { STATIC_DEMO } from '../lib/demo'
import { DemoBanner } from './DemoBanner'
import { Logo } from './Icon'

export function Layout() {
  return (
    <>
      <a
        href="#main"
        className="sr-only"
        onClick={
          STATIC_DEMO
            ? (e) => {
                e.preventDefault()
                document.getElementById('main')?.focus()
              }
            : undefined
        }
      >
        Skip to content
      </a>
      {STATIC_DEMO && <DemoBanner />}
      <header className="app-header">
        <div className="app-header__inner">
          <NavLink to="/" className="brand" aria-label="DataLens home">
            <Logo />
            DataLens
          </NavLink>
          <nav className="app-nav" aria-label="Primary">
            <NavLink to="/" end>
              Datasets
            </NavLink>
          </nav>
          <div className="app-header__spacer" />
          <span className="app-header__meta">Data quality for LLM fine-tuning datasets</span>
        </div>
      </header>
      <main id="main" tabIndex={STATIC_DEMO ? -1 : undefined}>
        <Outlet />
      </main>
    </>
  )
}
