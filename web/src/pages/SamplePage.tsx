import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { PiiSpan, SampleDetail } from '../api/types'
import { EmptyState } from '../components/EmptyState'
import { ErrorBanner } from '../components/ErrorBanner'
import { Icon } from '../components/Icon'
import { QCPill } from '../components/Pills'
import { QCResultsList } from '../components/sample/QCResultsList'
import { RichText } from '../components/sample/RichText'
import { SimilarSamples } from '../components/sample/SimilarSamples'
import { LoadingBlock, Spinner } from '../components/Spinner'
import { useDocumentTitle } from '../hooks/useDocumentTitle'
import { useResource } from '../hooks/useResource'
import { recallDatasetSearch } from '../lib/filter'
import { formatScore } from '../lib/format'
import { piiKindClass, piiKindLabel } from '../lib/pii'
import { piiSpans } from '../lib/text'

const RERUN_POLL_MS = 1500
const RERUN_MAX_POLLS = 10

const resultsKey = (s?: SampleDetail) =>
  s
    ? JSON.stringify(
        s.qc_results.map((r) => [r.check_name, r.severity, r.message]).sort((a, b) => a[0].localeCompare(b[0])),
      ) + s.qc_status
    : ''

export function SamplePage() {
  const { id } = useParams()
  const sampleId = Number(id)
  const valid = Number.isInteger(sampleId) && sampleId > 0
  const navigate = useNavigate()

  // after "Re-run QC", poll until the results change (or give up after a few tries)
  const rerun = useRef<{ before: string; polls: number } | null>(null)
  const [rerunning, setRerunning] = useState(false)

  const sample = useResource(valid ? () => api.getSample(sampleId) : null, [sampleId], (s) => {
    const r = rerun.current
    if (r) {
      r.polls += 1
      if ((s && resultsKey(s) !== r.before) || r.polls >= RERUN_MAX_POLLS) {
        rerun.current = null
        setRerunning(false)
        return false
      }
      return RERUN_POLL_MS
    }
    return s && s.qc_status === 'pending' ? 3000 : false
  })
  const s = sample.data?.id === sampleId ? sample.data : undefined
  const datasetId = s?.dataset_id
  const dataset = useResource(datasetId ? () => api.getDataset(datasetId) : null, [datasetId])

  useDocumentTitle(s ? `Sample #${s.sample_index}` : 'Sample')

  const [rerunError, setRerunError] = useState<unknown>(null)
  const rerunQC = async () => {
    if (!s) return
    setRerunError(null)
    setRerunning(true)
    try {
      await api.runSampleQC(s.id)
      rerun.current = { before: resultsKey(s), polls: 0 }
      sample.reload()
    } catch (e) {
      setRerunError(e)
      setRerunning(false)
    }
  }

  // keyboard shortcuts: [ / ] prev/next sample
  const prevId = s?.prev_id
  const nextId = s?.next_id
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable))
        return
      if (e.key === '[' && prevId) navigate(`/samples/${prevId}`)
      else if (e.key === ']' && nextId) navigate(`/samples/${nextId}`)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [prevId, nextId, navigate])

  const spans = useMemo(() => piiSpans(s?.qc_results), [s?.qc_results])
  const spansByField = useMemo(() => {
    const m: Record<string, PiiSpan[]> = {}
    for (const sp of spans) (m[sp.field] ??= []).push(sp)
    return m
  }, [spans])

  if (!valid) return <NotFound text="That sample id isn’t valid." />
  if (sample.initial && sample.loading) {
    return (
      <div className="page">
        <LoadingBlock label="Loading sample…" />
      </div>
    )
  }
  if (!s) {
    if (sample.error instanceof ApiError && sample.error.status === 404)
      return <NotFound text={`Sample #${sampleId} doesn’t exist or its dataset was deleted.`} />
    return (
      <div className="page">
        <nav className="breadcrumb" aria-label="Breadcrumb">
          <Link to="/">Datasets</Link>
        </nav>
        {sample.loading ? (
          <LoadingBlock label="Loading sample…" />
        ) : (
          <ErrorBanner title="Could not load sample" error={sample.error} onRetry={sample.reload} />
        )}
      </div>
    )
  }

  const ds = dataset.data?.id === s.dataset_id ? dataset.data : undefined
  const backTo = `/datasets/${s.dataset_id}${recallDatasetSearch(s.dataset_id)}`
  const kinds = Object.entries(
    spans.reduce<Record<string, number>>((acc, sp) => {
      acc[sp.kind] = (acc[sp.kind] ?? 0) + 1
      return acc
    }, {}),
  )
  const totalChars = s.prompt_chars + (s.context?.length ?? 0) + s.response_chars

  return (
    <div className="page">
      <nav className="breadcrumb" aria-label="Breadcrumb">
        <Link to="/">Datasets</Link>
        <span aria-hidden="true">/</span>
        <Link to={backTo} className="mono">
          {ds?.hf_repo_id ?? `Dataset #${s.dataset_id}`}
        </Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">Sample #{s.sample_index}</span>
      </nav>

      <div className="page-head">
        <div className="page-head__title">
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <h1>Sample #{s.sample_index}</h1>
            <QCPill status={s.qc_status} />
            {s.category && <span className="tag">{s.category}</span>}
          </div>
          <p className="secondary" style={{ margin: 0, fontSize: 13 }}>
            {s.tokens_est.toLocaleString()} tokens · QC score {formatScore(s.qc_score)}
            {sample.loading && !sample.initial && (
              <>
                {' '}
                <Spinner size={10} label="Refreshing" />
              </>
            )}
          </p>
        </div>
        <div className="page-head__actions">
          <button type="button" className="btn" onClick={rerunQC} disabled={rerunning}>
            {rerunning ? <Spinner size={12} /> : <Icon name="refresh" size={14} />}
            {rerunning ? 'Re-running…' : 'Re-run QC'}
          </button>
          <div className="sample-nav">
            <button
              type="button"
              className="btn"
              disabled={!s.prev_id}
              onClick={() => s.prev_id && navigate(`/samples/${s.prev_id}`)}
              aria-keyshortcuts="["
              title="Previous sample ( [ )"
            >
              <Icon name="chevronLeft" size={14} /> Prev <kbd className="hide-xs">[</kbd>
            </button>
            <button
              type="button"
              className="btn"
              disabled={!s.next_id}
              onClick={() => s.next_id && navigate(`/samples/${s.next_id}`)}
              aria-keyshortcuts="]"
              title="Next sample ( ] )"
            >
              Next <kbd className="hide-xs">]</kbd> <Icon name="chevronRight" size={14} />
            </button>
          </div>
        </div>
      </div>

      {rerunError != null && <ErrorBanner title="Could not re-run QC" error={rerunError} />}
      {sample.error != null && (
        <ErrorBanner title="Could not refresh sample" error={sample.error} onRetry={sample.reload} />
      )}

      <div className="viewer">
        <div className="stack">
          <FieldCard title="Prompt" chars={s.prompt_chars} text={s.prompt} spans={spansByField.prompt} />
          {s.context != null && s.context.trim() !== '' && (
            <FieldCard title="Context" chars={s.context.length} text={s.context} spans={spansByField.context} />
          )}
          <FieldCard title="Response" chars={s.response_chars} text={s.response} spans={spansByField.response} />
        </div>
        <div className="stack sticky-col">
          <section className="card" aria-labelledby="meta-heading">
            <div className="card__head">
              <h2 id="meta-heading">Metadata</h2>
            </div>
            <dl className="meta-grid">
              <div>
                <dt>Category</dt>
                <dd>{s.category ?? <span className="muted">—</span>}</dd>
              </div>
              <div>
                <dt>Language</dt>
                <dd className="mono">{s.lang ?? '—'}</dd>
              </div>
              <div>
                <dt>Tokens (est.)</dt>
                <dd className="num">{s.tokens_est.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Characters</dt>
                <dd className="num">{totalChars.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Prompt / response</dt>
                <dd className="num">
                  {s.prompt_chars.toLocaleString()} / {s.response_chars.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt>QC score</dt>
                <dd className="num">{formatScore(s.qc_score)}</dd>
              </div>
              <div>
                <dt>Row index</dt>
                <dd className="num">{s.sample_index}</dd>
              </div>
              <div>
                <dt>Sample id</dt>
                <dd className="num mono">{s.id}</dd>
              </div>
            </dl>
          </section>

          {kinds.length > 0 && (
            <section className="card" aria-labelledby="pii-heading">
              <div className="card__head">
                <div>
                  <h2 id="pii-heading">PII highlighted</h2>
                  <p>Detected spans are marked inline in the text.</p>
                </div>
              </div>
              <ul className="pii-legend">
                {kinds.map(([k, n]) => (
                  <li key={k}>
                    <mark className={`pii ${piiKindClass(k)}`}>{piiKindLabel(k)}</mark>
                    <span className="num muted">× {n}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section className="card" aria-labelledby="qc-heading">
            <div className="card__head">
              <div>
                <h2 id="qc-heading">Quality checks</h2>
                <p>
                  {s.qc_results.length
                    ? `${s.qc_results.filter((r) => !r.passed).length} of ${s.qc_results.length} flagged`
                    : ' '}
                </p>
              </div>
            </div>
            <QCResultsList results={s.qc_results} pending={s.qc_status === 'pending'} />
          </section>

          <SimilarSamples sampleId={s.id} />
        </div>
      </div>
    </div>
  )
}

function FieldCard({
  title,
  text,
  chars,
  spans,
}: {
  title: string
  text: string
  chars: number
  spans?: PiiSpan[]
}) {
  return (
    <section className="card field-card" aria-label={title}>
      <div className="card__head">
        <h2>{title}</h2>
        <span className="muted num" style={{ fontSize: 12 }}>
          {chars.toLocaleString()} chars
          {spans?.length ? ` · ${spans.length} PII` : ''}
        </span>
      </div>
      <div className="card__body">
        <RichText text={text} spans={spans} />
      </div>
    </section>
  )
}

function NotFound({ text }: { text: string }) {
  return (
    <div className="page">
      <nav className="breadcrumb" aria-label="Breadcrumb">
        <Link to="/">Datasets</Link>
      </nav>
      <div className="card">
        <EmptyState
          icon="search"
          title="Sample not found"
          actions={
            <Link to="/" className="btn">
              Back to datasets
            </Link>
          }
        >
          {text}
        </EmptyState>
      </div>
    </div>
  )
}
