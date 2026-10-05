import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { DatasetDetail, DatasetOut, FieldMap, QCCounts } from '../api/types'
import { ConfirmButton } from '../components/ConfirmButton'
import { EmptyState } from '../components/EmptyState'
import { ErrorBanner } from '../components/ErrorBanner'
import { Icon } from '../components/Icon'
import { StatusPill } from '../components/Pills'
import { QCDistributionBar } from '../components/dataset/QCSummaryCards'
import { LoadingBlock, Spinner } from '../components/Spinner'
import { useDocumentTitle } from '../hooks/useDocumentTitle'
import { useResource } from '../hooks/useResource'
import { formatDate, relativeTime } from '../lib/format'
import { isBusyStatus } from '../lib/status'

const REPO_RE = /^[\w.-]+\/[\w.-]+$/
const POLL_MS = 2000
const QC_POLL_MS = 4000

const EXAMPLES: { repo: string; config?: string; note: string }[] = [
  { repo: 'databricks/databricks-dolly-15k', note: 'instruction / context / response' },
  { repo: 'tatsu-lab/alpaca', note: 'instruction / input / output' },
  { repo: 'HuggingFaceH4/no_robots', note: 'chat messages' },
  { repo: 'openai/gsm8k', config: 'main', note: 'math question / answer' },
]

const FIELD_KEYS: (keyof FieldMap)[] = ['prompt', 'context', 'response', 'category', 'messages']

export function DatasetsPage() {
  useDocumentTitle('Datasets')
  const list = useResource(() => api.listDatasets(), [])
  const datasets = list.data ?? []
  const setDatasets = list.setData
  const [details, setDetails] = useState<Record<number, DatasetDetail>>({})
  const [rowError, setRowError] = useState<string | null>(null)

  // Poll GET /datasets/{id} every 2 s while pending/importing.
  const busyKey = datasets
    .filter((d) => isBusyStatus(d.status))
    .map((d) => d.id)
    .join(',')
  useEffect(() => {
    if (!busyKey) return
    const ids = busyKey.split(',').map(Number)
    const t = setInterval(async () => {
      const results = await Promise.allSettled(ids.map((id) => api.getDataset(id)))
      const fresh: Record<number, DatasetDetail> = {}
      results.forEach((r, i) => {
        if (r.status === 'fulfilled') fresh[ids[i]] = r.value
      })
      setDetails((prev) => ({ ...prev, ...fresh }))
      setDatasets((prev) => prev?.map((d) => (fresh[d.id] ? { ...d, ...fresh[d.id] } : d)))
    }, POLL_MS)
    return () => clearInterval(t)
  }, [busyKey, setDatasets])

  // QC counts for ready datasets (the list endpoint has none) — refreshed while QC is still running.
  const readyKey = datasets
    .filter((d) => d.status === 'ready')
    .map((d) => d.id)
    .join(',')
  const qcRunningKey = Object.values(details)
    .filter((d) => d.status === 'ready' && d.qc_counts.pending > 0)
    .map((d) => d.id)
    .join(',')
  useEffect(() => {
    if (!readyKey) return
    let cancelled = false
    const load = async (ids: number[]) => {
      const results = await Promise.allSettled(ids.map((id) => api.getDataset(id)))
      if (cancelled) return
      const fresh: Record<number, DatasetDetail> = {}
      results.forEach((r, i) => {
        if (r.status === 'fulfilled') fresh[ids[i]] = r.value
      })
      setDetails((prev) => ({ ...prev, ...fresh }))
    }
    load(readyKey.split(',').map(Number))
    return () => {
      cancelled = true
    }
  }, [readyKey])
  useEffect(() => {
    if (!qcRunningKey) return
    const ids = qcRunningKey.split(',').map(Number)
    const t = setInterval(async () => {
      const results = await Promise.allSettled(ids.map((id) => api.getDataset(id)))
      const fresh: Record<number, DatasetDetail> = {}
      results.forEach((r, i) => {
        if (r.status === 'fulfilled') fresh[ids[i]] = r.value
      })
      setDetails((prev) => ({ ...prev, ...fresh }))
    }, QC_POLL_MS)
    return () => clearInterval(t)
  }, [qcRunningKey])

  const onImported = (d: DatasetOut) => {
    setDatasets((prev) => [d, ...(prev ?? []).filter((p) => p.id !== d.id)])
  }

  const onDelete = async (id: number) => {
    setRowError(null)
    try {
      await api.deleteDataset(id)
      setDatasets((prev) => prev?.filter((d) => d.id !== id))
    } catch (e) {
      setRowError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <div className="page-head__title">
          <h1>Datasets</h1>
          <p className="secondary" style={{ margin: 0 }}>
            Import instruction and chat datasets from the Hugging Face Hub. Every sample gets automatic data-quality
            checks — duplicates, PII, refusals, language, formatting and length.
          </p>
        </div>
      </div>

      <ImportForm onImported={onImported} />

      <section className="card" aria-labelledby="ds-heading">
        <div className="card__head">
          <div>
            <h2 id="ds-heading">Imported datasets</h2>
            <p>{list.data ? `${datasets.length} total` : ' '}</p>
          </div>
          <button
            type="button"
            className="btn btn--sm"
            onClick={list.reload}
            disabled={list.loading}
            aria-label="Refresh dataset list"
          >
            {list.loading ? <Spinner size={11} /> : <Icon name="refresh" size={13} />}
            <span className="hide-xs">Refresh</span>
          </button>
        </div>
        {rowError && (
          <div className="card__body" style={{ paddingBottom: 0 }}>
            <ErrorBanner title="Could not delete dataset" error={rowError} />
          </div>
        )}
        {list.initial && list.loading ? (
          <LoadingBlock label="Loading datasets…" />
        ) : list.error && !list.data ? (
          <div className="card__body">
            <ErrorBanner title="Could not load datasets" error={list.error} onRetry={list.reload} />
          </div>
        ) : datasets.length === 0 ? (
          <EmptyState icon="database" title="No datasets yet">
            Pick one of the example datasets above — <code>databricks/databricks-dolly-15k</code> is a good start — or
            paste any public Hugging Face repo id. DataLens imports the first N rows, detects the prompt/response
            columns and runs quality checks on every sample.
          </EmptyState>
        ) : (
          <DatasetTable datasets={datasets} details={details} onDelete={onDelete} />
        )}
      </section>
    </div>
  )
}

function ImportForm({ onImported }: { onImported: (d: DatasetOut) => void }) {
  const [repo, setRepo] = useState('databricks/databricks-dolly-15k')
  const [config, setConfig] = useState('')
  const [split, setSplit] = useState('train')
  const [maxSamples, setMaxSamples] = useState('2000')
  const [fields, setFields] = useState<Record<string, string>>({})
  const [touched, setTouched] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const uid = useId()

  const repoErr = !REPO_RE.test(repo.trim()) ? 'Use the form owner/name, e.g. databricks/databricks-dolly-15k' : null
  const n = Number(maxSamples)
  const maxErr = !Number.isInteger(n) || n < 1 || n > 20000 ? 'Enter a whole number from 1 to 20,000' : null
  const splitErr = !split.trim() ? 'Split is required' : null

  const fieldMap: FieldMap = {}
  for (const k of FIELD_KEYS) if (fields[k]?.trim()) fieldMap[k] = fields[k].trim()
  const hasFields = Object.keys(fieldMap).length > 0

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setTouched(true)
    setError(null)
    setSuccess(null)
    if (repoErr || maxErr || splitErr) return
    setSubmitting(true)
    try {
      const d = await api.importDataset({
        hf_repo_id: repo.trim(),
        config: config.trim() || null,
        split: split.trim(),
        max_samples: n,
        fields: hasFields ? fieldMap : null,
      })
      onImported(d)
      setSuccess(`Import of ${d.hf_repo_id} queued.`)
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 409
          ? `${repo.trim()} (${config.trim() || 'default'}/${split.trim()}) is already imported. Delete it first to re-import.`
          : err instanceof Error
            ? err.message
            : String(err),
      )
    } finally {
      setSubmitting(false)
    }
  }

  const pick = (ex: (typeof EXAMPLES)[number]) => {
    setRepo(ex.repo)
    setConfig(ex.config ?? '')
    setSplit('train')
    setError(null)
    setSuccess(null)
  }

  return (
    <section className="card" aria-labelledby="import-heading">
      <div className="card__head">
        <div>
          <h2 id="import-heading">Import dataset</h2>
          <p>Any public instruction or chat dataset on the Hugging Face Hub (parquet or JSONL).</p>
        </div>
      </div>
      <form className="card__body" onSubmit={submit} noValidate>
        <div className="examples" style={{ marginBottom: 12 }} role="group" aria-label="Example datasets">
          <span>Examples:</span>
          {EXAMPLES.map((ex) => {
            const active = repo.trim() === ex.repo && (config.trim() || undefined) === ex.config
            return (
              <button
                key={ex.repo}
                type="button"
                className="example-btn mono"
                aria-pressed={active}
                title={ex.note}
                onClick={() => pick(ex)}
              >
                {ex.repo}
                {ex.config && <span className="muted"> · {ex.config}</span>}
              </button>
            )
          })}
        </div>
        <div className="form-row">
          <div className="field" style={{ flex: '2 1 260px' }}>
            <label htmlFor={`${uid}-repo`}>Hugging Face repo id</label>
            <input
              id={`${uid}-repo`}
              className="input mono"
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              onBlur={() => setTouched(true)}
              placeholder="owner/name"
              autoComplete="off"
              spellCheck={false}
              aria-invalid={touched && !!repoErr}
              aria-describedby={`${uid}-msg`}
            />
          </div>
          <div className="field field--narrow">
            <label htmlFor={`${uid}-config`}>Config</label>
            <input
              id={`${uid}-config`}
              className="input mono"
              value={config}
              onChange={(e) => setConfig(e.target.value)}
              placeholder="default"
              autoComplete="off"
              spellCheck={false}
            />
          </div>
          <div className="field field--narrow">
            <label htmlFor={`${uid}-split`}>Split</label>
            <input
              id={`${uid}-split`}
              className="input mono"
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              aria-invalid={touched && !!splitErr}
            />
          </div>
          <div className="field field--narrow">
            <label htmlFor={`${uid}-max`}>Max samples</label>
            <input
              id={`${uid}-max`}
              className="input num"
              type="number"
              min={1}
              max={20000}
              step={1}
              value={maxSamples}
              onChange={(e) => setMaxSamples(e.target.value)}
              aria-invalid={touched && !!maxErr}
              aria-describedby={`${uid}-msg`}
            />
          </div>
          <button type="submit" className="btn btn--primary" disabled={submitting}>
            {submitting ? <Spinner size={12} /> : <Icon name="download" size={14} />}
            Import
          </button>
        </div>

        <details className="disclosure">
          <summary>
            Advanced: field mapping
            {hasFields && <span className="pill pill--info" style={{ marginLeft: 8 }}>{Object.keys(fieldMap).length} set</span>}
          </summary>
          <p className="field__hint" style={{ margin: '0 0 10px' }}>
            Leave blank to auto-detect (e.g. <code>instruction</code> → prompt, <code>output</code> → response). For chat
            datasets, set only the messages column.
          </p>
          <div className="form-row">
            {FIELD_KEYS.map((k) => (
              <div className="field" key={k} style={{ flex: '1 1 140px' }}>
                <label htmlFor={`${uid}-f-${k}`}>{k === 'messages' ? 'Messages (chat)' : k[0].toUpperCase() + k.slice(1)} column</label>
                <input
                  id={`${uid}-f-${k}`}
                  className="input mono"
                  value={fields[k] ?? ''}
                  onChange={(e) => setFields((f) => ({ ...f, [k]: e.target.value }))}
                  placeholder="auto"
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
            ))}
          </div>
        </details>

        <div id={`${uid}-msg`} style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: '2px 12px' }}>
          {touched && repoErr && <span className="field__error">{repoErr}</span>}
          {touched && splitErr && <span className="field__error">{splitErr}</span>}
          {touched && maxErr && <span className="field__error">Max samples: {maxErr}</span>}
          {success && !error && (
            <span className="field__hint" role="status">
              {success} Progress updates below.
            </span>
          )}
        </div>
        {error && <ErrorBanner title="Import failed" error={error} />}
      </form>
    </section>
  )
}

function ImportProgress({ d }: { d: DatasetOut }) {
  const total = d.total_samples
  const pct = total ? Math.min(100, (d.num_samples / total) * 100) : 0
  const indeterminate = d.status === 'pending' || !total || d.num_samples === 0
  return (
    <span
      className={`progress${indeterminate ? ' progress--indeterminate' : ''}`}
      role="progressbar"
      aria-label={`Import progress for ${d.hf_repo_id}`}
      aria-valuemin={0}
      aria-valuemax={total ?? undefined}
      aria-valuenow={indeterminate ? undefined : d.num_samples}
    >
      <span style={indeterminate ? undefined : { width: `${pct}%` }} />
    </span>
  )
}

function PassRate({ counts }: { counts?: QCCounts }) {
  if (!counts) return <span className="muted">—</span>
  const checked = counts.pass + counts.warn + counts.fail
  const total = checked + counts.pending
  if (!checked) {
    return (
      <span className="muted" style={{ fontSize: 12 }}>
        {total ? 'QC queued' : '—'}
      </span>
    )
  }
  return (
    <div className="pass-rate">
      <span className="num pass-rate__value">{Math.round((counts.pass / total) * 100)}%</span>
      <QCDistributionBar counts={counts} />
    </div>
  )
}

function DatasetTable({
  datasets,
  details,
  onDelete,
}: {
  datasets: DatasetOut[]
  details: Record<number, DatasetDetail>
  onDelete: (id: number) => Promise<void>
}) {
  const navigate = useNavigate()
  const bodyRef = useRef<HTMLTableSectionElement>(null)

  const onRowKey = (e: KeyboardEvent<HTMLTableRowElement>, id: number) => {
    if (e.target !== e.currentTarget) return
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      navigate(`/datasets/${id}`)
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      const rows = Array.from(bodyRef.current?.querySelectorAll<HTMLTableRowElement>('tr.row-link') ?? [])
      const i = rows.indexOf(e.currentTarget)
      rows[Math.max(0, Math.min(rows.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))]?.focus()
    }
  }

  return (
    <div className="table-wrap">
      <table className="table">
        <caption className="sr-only">Imported datasets. Press Enter on a row to open it.</caption>
        <thead>
          <tr>
            <th scope="col">Repository</th>
            <th scope="col" className="hide-sm">
              Config / split
            </th>
            <th scope="col">Status</th>
            <th scope="col" className="r">
              Samples
            </th>
            <th scope="col" className="hide-xs" style={{ minWidth: 130 }}>
              QC pass rate
            </th>
            <th scope="col" className="hide-sm">
              Created
            </th>
            <th scope="col">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody ref={bodyRef}>
          {datasets.map((d) => {
            const detail = details[d.id]
            return (
              <tr
                key={d.id}
                className="row-link"
                tabIndex={0}
                onClick={() => navigate(`/datasets/${d.id}`)}
                onKeyDown={(e) => onRowKey(e, d.id)}
                aria-label={`Open ${d.hf_repo_id}`}
              >
                <td>
                  <div className="cell-title mono" style={{ overflowWrap: 'anywhere' }}>
                    {d.hf_repo_id}
                  </div>
                  <div className="cell-sub mono show-sm">
                    {d.config} / {d.split}
                  </div>
                  {d.status === 'failed' && d.error && (
                    <div className="cell-sub" style={{ color: 'var(--fail)', overflowWrap: 'anywhere' }}>
                      {d.error}
                    </div>
                  )}
                </td>
                <td className="hide-sm">
                  <span className="mono secondary" style={{ fontSize: 12.5 }}>
                    {d.config} / {d.split}
                  </span>
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <StatusPill status={d.status} title={d.error ?? undefined} />
                    {isBusyStatus(d.status) && <ImportProgress d={d} />}
                  </div>
                </td>
                <td className="r num nowrap">
                  {d.num_samples.toLocaleString()}
                  <span className="muted"> / {d.total_samples != null ? d.total_samples.toLocaleString() : '?'}</span>
                </td>
                <td className="hide-xs">
                  {d.status === 'ready' ? <PassRate counts={detail?.qc_counts} /> : <span className="muted">—</span>}
                </td>
                <td className="hide-sm" title={formatDate(d.created_at)}>
                  <span className="secondary">{relativeTime(d.created_at) || formatDate(d.created_at)}</span>
                </td>
                <td className="r">
                  <ConfirmButton
                    small
                    label="Delete"
                    ariaLabel={`Delete ${d.hf_repo_id}`}
                    prompt="Delete?"
                    onConfirm={() => onDelete(d.id)}
                  />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
