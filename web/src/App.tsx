import { useEffect, useState } from 'react'
import './App.css'

type Check = { label: string; path: string; state: 'loading' | 'ok' | 'down'; detail: string }

const CHECKS: Pick<Check, 'label' | 'path'>[] = [
  { label: 'API', path: '/api/health' },
  { label: 'Database', path: '/api/health/ready' },
]

// Dataset shape returned by GET /datasets once Week 3 is implemented.
type Dataset = { id: number; hf_repo_id: string; status: string; num_episodes: number | null }

type DatasetsState =
  | { kind: 'loading' }
  | { kind: 'ready'; items: Dataset[] }
  | { kind: 'todo'; message: string }
  | { kind: 'error'; message: string }

export default function App() {
  const [checks, setChecks] = useState<Check[]>(CHECKS.map((c) => ({ ...c, state: 'loading', detail: '' })))
  const [datasets, setDatasets] = useState<DatasetsState>({ kind: 'loading' })

  useEffect(() => {
    CHECKS.forEach((c, i) => {
      fetch(c.path)
        .then(async (r) => {
          const body = await r.json().catch(() => ({}))
          const next: Check = { ...c, state: r.ok ? 'ok' : 'down', detail: body.status ?? body.detail ?? `HTTP ${r.status}` }
          setChecks((prev) => prev.map((p, j) => (j === i ? next : p)))
        })
        .catch(() => setChecks((prev) => prev.map((p, j) => (j === i ? { ...c, state: 'down', detail: 'unreachable' } : p))))
    })

    fetch('/api/datasets')
      .then(async (r) => {
        const body = await r.json().catch(() => ({}))
        if (r.status === 501) setDatasets({ kind: 'todo', message: body.detail ?? 'Not implemented yet' })
        else if (r.ok) setDatasets({ kind: 'ready', items: body as Dataset[] })
        else setDatasets({ kind: 'error', message: body.detail ?? `HTTP ${r.status}` })
      })
      .catch(() => setDatasets({ kind: 'error', message: 'API unreachable. Is docker compose running?' }))
  }, [])

  return (
    <main className="page">
      <header className="top">
        <h1>EpisodeHub</h1>
        <p className="muted">Import, quality-check and search robot-learning datasets.</p>
      </header>

      <section className="card">
        <h2>System status</h2>
        <div className="checks">
          {checks.map((c) => (
            <div key={c.path} className={`check ${c.state}`}>
              <span className="dot" />
              <span className="label">{c.label}</span>
              <span className="muted">{c.state === 'loading' ? 'checking…' : c.detail}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Datasets</h2>
        {datasets.kind === 'loading' && <p className="muted">Loading…</p>}
        {datasets.kind === 'todo' && (
          <p className="note">
            <b>{datasets.message}.</b> Implement <code>GET /datasets</code> and <code>POST /datasets/import</code> in{' '}
            <code>api/app/routers/datasets.py</code>, then this list fills in.
          </p>
        )}
        {datasets.kind === 'error' && <p className="error">{datasets.message}</p>}
        {datasets.kind === 'ready' &&
          (datasets.items.length ? (
            <table>
              <thead>
                <tr><th>Repo</th><th>Status</th><th>Episodes</th></tr>
              </thead>
              <tbody>
                {datasets.items.map((d) => (
                  <tr key={d.id}><td>{d.hf_repo_id}</td><td>{d.status}</td><td>{d.num_episodes ?? '–'}</td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No datasets yet. Import lerobot/pusht to get started.</p>
          ))}
      </section>

      <section className="card">
        <h2>Coming next</h2>
        <ul className="todo">
          <li><b>Week 3</b> Import a Hugging Face LeRobot dataset</li>
          <li><b>Week 4</b> Quality-check results per episode</li>
          <li><b>Week 5</b> Episode browser with video synced to joint-state plots</li>
          <li><b>Week 6</b> Natural-language search and export</li>
        </ul>
      </section>
    </main>
  )
}
