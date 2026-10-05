import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api, ApiError, filterToParams } from '../api/client'
import type { QCSummary, SampleFilter } from '../api/types'
import { CategoryBars } from '../components/dataset/CategoryBars'
import { CheckBreakdown } from '../components/dataset/CheckBreakdown'
import { ExportPanel } from '../components/dataset/ExportPanel'
import { QCDistributionBar, QCSummaryCards } from '../components/dataset/QCSummaryCards'
import { SamplesTable } from '../components/dataset/SamplesTable'
import { SearchPanel, type SearchMeta } from '../components/dataset/SearchPanel'
import { TokenHistogram } from '../components/dataset/TokenHistogram'
import { EmptyState } from '../components/EmptyState'
import { ErrorBanner } from '../components/ErrorBanner'
import { Icon } from '../components/Icon'
import { StatusPill } from '../components/Pills'
import { LoadingBlock, Spinner } from '../components/Spinner'
import { useDocumentTitle } from '../hooks/useDocumentTitle'
import { useResource } from '../hooks/useResource'
import { cleanFilter, filterFromParams, isEmptyFilter, rememberDatasetSearch } from '../lib/filter'
import { formatDate } from '../lib/format'
import { isBusyStatus } from '../lib/status'

const PAGE_SIZE = 25
const POLL_MS = 2000

export function DatasetDetailPage() {
  const { id } = useParams()
  const datasetId = Number(id)
  const valid = Number.isInteger(datasetId) && datasetId > 0

  // ---- URL-backed state: filter, page, search meta ----
  const [params, setParams] = useSearchParams()
  const paramsKey = params.toString()
  const filter = useMemo(() => filterFromParams(new URLSearchParams(paramsKey)), [paramsKey])
  const offset = Math.max(0, Number(params.get('offset')) || 0)
  const q = params.get('q')
  const parser = params.get('parser')
  const searchMeta: SearchMeta | null = useMemo(
    () => (q ? { query: q, parser: parser === 'llm' ? 'llm' : 'rules' } : null),
    [q, parser],
  )

  useEffect(() => {
    if (valid) rememberDatasetSearch(datasetId, paramsKey ? `?${paramsKey}` : '')
  }, [valid, datasetId, paramsKey])

  const writeParams = useCallback(
    (f: SampleFilter, meta: SearchMeta | null, nextOffset = 0) => {
      const p = new URLSearchParams(filterToParams(cleanFilter(f)).replace(/^\?/, ''))
      if (meta) {
        p.set('q', meta.query)
        p.set('parser', meta.parser)
      }
      if (nextOffset) p.set('offset', String(nextOffset))
      setParams(p, { replace: true })
    },
    [setParams],
  )

  const setFilter = (f: SampleFilter, keepSearch = false) => writeParams(f, keepSearch ? searchMeta : null)

  // ---- data ----
  const dataset = useResource(valid ? () => api.getDataset(datasetId) : null, [datasetId], (d) =>
    d && isBusyStatus(d.status) ? POLL_MS : false,
  )
  const ds = dataset.data?.id === datasetId ? dataset.data : undefined
  useDocumentTitle(ds?.hf_repo_id ?? 'Dataset')

  const [qcPolling, setQcPolling] = useState(0)
  const summary = useResource<QCSummary>(
    valid ? () => api.getQCSummary(datasetId) : null,
    [datasetId, ds?.status, qcPolling],
    (s) => (s && s.counts.pending > 0 ? POLL_MS : false),
  )
  const counts = summary.data?.counts ?? ds?.qc_counts
  const countsKey = counts ? `${counts.pass}-${counts.warn}-${counts.fail}-${counts.pending}` : ''

  const samples = useResource(
    valid ? () => api.listSamples(datasetId, filter, PAGE_SIZE, offset) : null,
    // refetch when filter/page change, when import progresses, and when QC counts move
    [datasetId, paramsKey, offset, ds?.num_samples, ds?.status, countsKey],
  )

  const [qcBusy, setQcBusy] = useState(false)
  const [qcError, setQcError] = useState<unknown>(null)
  const runQC = async () => {
    setQcBusy(true)
    setQcError(null)
    try {
      await api.runDatasetQC(datasetId)
      setQcPolling((n) => n + 1)
    } catch (e) {
      setQcError(e)
    } finally {
      setQcBusy(false)
    }
  }

  if (!valid) return <NotFoundCard text="That dataset id isn’t valid." />

  if (dataset.initial && dataset.loading) {
    return (
      <div className="page">
        <LoadingBlock label="Loading dataset…" />
      </div>
    )
  }
  if (!ds) {
    if (dataset.error instanceof ApiError && dataset.error.status === 404)
      return <NotFoundCard text={`Dataset #${datasetId} doesn’t exist or was deleted.`} />
    return (
      <div className="page">
        <Breadcrumb />
        <ErrorBanner title="Could not load dataset" error={dataset.error} onRetry={dataset.reload} />
      </div>
    )
  }

  const busy = isBusyStatus(ds.status)
  const qcRunning = (counts?.pending ?? 0) > 0 && ds.status === 'ready'
  const filtered = !isEmptyFilter(filter)
  const fieldEntries = Object.entries(ds.fields ?? {}).filter(
    (e): e is [string, string] => typeof e[1] === 'string' && e[1] !== '',
  )
  const checked = counts ? counts.pass + counts.warn + counts.fail : 0

  return (
    <div className="page">
      <Breadcrumb name={ds.hf_repo_id} />
      <div className="page-head">
        <div className="page-head__title">
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <h1 className="mono">
              <a
                className="title-link"
                href={`https://huggingface.co/datasets/${ds.hf_repo_id}`}
                target="_blank"
                rel="noreferrer"
                title="Open on Hugging Face"
              >
                {ds.hf_repo_id}
                <Icon name="external" size={14} style={{ marginLeft: 6, verticalAlign: '1px' }} />
                <span className="sr-only"> (opens Hugging Face in a new tab)</span>
              </a>
            </h1>
            <StatusPill status={ds.status} />
          </div>
          <dl className="meta-row" style={{ margin: 0 }}>
            <div>
              <dt>Config</dt>
              <dd className="mono">{ds.config}</dd>
            </div>
            <div>
              <dt>Split</dt>
              <dd className="mono">{ds.split}</dd>
            </div>
            <div>
              <dt>Samples</dt>
              <dd className="num">
                {ds.num_samples.toLocaleString()}
                {ds.total_samples != null && <span className="muted"> of {ds.total_samples.toLocaleString()}</span>}
              </dd>
            </div>
            {ds.revision && (
              <div>
                <dt>Revision</dt>
                <dd className="mono" title={ds.revision}>
                  {ds.revision.slice(0, 7)}
                </dd>
              </div>
            )}
            <div>
              <dt>Imported</dt>
              <dd>{formatDate(ds.created_at)}</dd>
            </div>
          </dl>
          {fieldEntries.length > 0 && (
            <div className="field-map" aria-label="Detected field mapping">
              <span className="muted">Fields</span>
              {fieldEntries.map(([role, col]) => (
                <span key={role} className="field-chip" title={`${role} ← column “${col}”`}>
                  <b>{role}</b>
                  <span aria-hidden="true">←</span>
                  <span className="sr-only">from column</span>
                  <code>{col}</code>
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="page-head__actions">
          <button type="button" className="btn" onClick={runQC} disabled={qcBusy || busy || ds.status === 'failed'}>
            {qcBusy || qcRunning ? <Spinner size={12} /> : <Icon name="refresh" size={14} />}
            {qcRunning ? 'QC running…' : 'Run QC again'}
          </button>
        </div>
      </div>

      {ds.status === 'failed' && <ErrorBanner title="Import failed" error={ds.error ?? 'Unknown error'} />}
      {busy && (
        <ErrorBanner tone="info" title={ds.status === 'pending' ? 'Import queued' : 'Importing samples…'}>
          {ds.num_samples.toLocaleString()} sample{ds.num_samples === 1 ? '' : 's'} imported so far. This page updates
          automatically.
        </ErrorBanner>
      )}
      {qcRunning && !busy && (
        <ErrorBanner tone="info" title="Quality checks running">
          {checked.toLocaleString()} of {(checked + (counts?.pending ?? 0)).toLocaleString()} samples checked. Results
          update live.
        </ErrorBanner>
      )}
      {qcError != null && <ErrorBanner title="Could not start QC" error={qcError} />}

      <QCSummaryCards counts={counts} loading={summary.loading && !counts} />
      {counts && <QCDistributionBar counts={counts} />}

      <div className="insights">
        <section className="card" aria-labelledby="checks-heading">
          <div className="card__head">
            <div>
              <h2 id="checks-heading">Checks flagging samples</h2>
              <p>Click a check to filter samples.</p>
            </div>
            {summary.loading && !summary.initial && <Spinner size={12} label="Updating QC summary" />}
          </div>
          <div className="card__body">
            {summary.data ? (
              <CheckBreakdown
                summary={summary.data}
                totalSamples={ds.num_samples}
                activeCheck={filter.failed_check}
                onSelect={(c) => setFilter({ ...filter, failed_check: c ?? undefined }, true)}
              />
            ) : summary.error != null ? (
              <ErrorBanner title="QC summary unavailable" error={summary.error} onRetry={summary.reload} />
            ) : (
              <LoadingBlock label="Loading QC summary…" />
            )}
          </div>
        </section>
        <section className="card" aria-labelledby="cat-heading">
          <div className="card__head">
            <div>
              <h2 id="cat-heading">Categories</h2>
              <p>
                {ds.categories.length
                  ? `${ds.categories.length} categor${ds.categories.length === 1 ? 'y' : 'ies'} · click to filter`
                  : 'Task category per sample'}
              </p>
            </div>
          </div>
          <div className="card__body">
            <CategoryBars
              categories={ds.categories}
              active={filter.category}
              onSelect={(c) => setFilter({ ...filter, category: c ?? undefined }, true)}
            />
          </div>
        </section>
        <section className="card" aria-labelledby="hist-heading">
          <div className="card__head">
            <div>
              <h2 id="hist-heading">Token length</h2>
              <p>Distribution of estimated tokens per sample.</p>
            </div>
          </div>
          <div className="card__body">
            <TokenHistogram
              buckets={ds.token_histogram}
              activeRange={{ min: filter.min_tokens, max: filter.max_tokens }}
              onSelect={(r) =>
                setFilter({ ...filter, min_tokens: r ? r.min : undefined, max_tokens: r ? r.max : undefined }, true)
              }
            />
          </div>
        </section>
      </div>

      <div className="grid-2">
        <div className="stack">
          <SearchPanel
            key={datasetId}
            datasetId={datasetId}
            filter={filter}
            categories={ds.categories}
            onFilterChange={setFilter}
            searchMeta={searchMeta}
            onSearch={(meta, f) => writeParams(f, meta)}
          />
          <SamplesTable
            data={samples.data}
            error={samples.error}
            loading={samples.loading}
            initial={samples.initial}
            offset={offset}
            pageSize={PAGE_SIZE}
            onPage={(o) => writeParams(filter, searchMeta, o)}
            onRetry={samples.reload}
            filtered={filtered}
            onClearFilter={() => writeParams({}, null)}
            importing={busy}
          />
        </div>
        <div className="stack sticky-col">
          <ExportPanel
            datasetId={datasetId}
            filter={filter}
            matching={samples.data?.total}
            disabled={ds.status !== 'ready'}
          />
        </div>
      </div>
    </div>
  )
}

function Breadcrumb({ name }: { name?: string }) {
  return (
    <nav className="breadcrumb" aria-label="Breadcrumb">
      <Link to="/">Datasets</Link>
      {name && (
        <>
          <span aria-hidden="true">/</span>
          <span aria-current="page" className="mono">
            {name}
          </span>
        </>
      )}
    </nav>
  )
}

function NotFoundCard({ text }: { text: string }) {
  return (
    <div className="page">
      <Breadcrumb />
      <div className="card">
        <EmptyState
          icon="database"
          title="Dataset not found"
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
