import { useId, useState } from 'react'
import { api } from '../../api/client'
import type { ExportFormat, ExportOut, SampleFilter } from '../../api/types'
import { useResource } from '../../hooks/useResource'
import { STATIC_DEMO } from '../../lib/demo'
import { describeFilter } from '../../lib/filter'
import { formatDate, plural, relativeTime } from '../../lib/format'
import { isBusyStatus } from '../../lib/status'
import { EmptyState } from '../EmptyState'
import { ErrorBanner } from '../ErrorBanner'
import { Icon } from '../Icon'
import { StatusPill } from '../Pills'
import { ReadOnlyNote } from '../ReadOnlyNote'
import { LoadingBlock, Spinner } from '../Spinner'

const FORMATS: { value: ExportFormat; label: string; hint: string }[] = [
  { value: 'jsonl', label: 'JSONL', hint: '{"prompt", "context", "response", "category"} per line' },
  { value: 'chat', label: 'Chat', hint: '{"messages": [user, assistant]} per line' },
]

export function ExportPanel({
  datasetId,
  filter,
  matching,
  disabled,
}: {
  datasetId: number
  filter: SampleFilter
  matching?: number
  disabled?: boolean
}) {
  const [valRatio, setValRatio] = useState(0.1)
  const [format, setFormat] = useState<ExportFormat>('jsonl')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<unknown>(null)
  const sliderId = useId()

  const exports = useResource<ExportOut[]>(
    () => api.listExports(datasetId),
    [datasetId],
    (data) => (data?.some((x) => isBusyStatus(x.status)) ? 2000 : false),
  )

  const create = async () => {
    setCreating(true)
    setCreateError(null)
    try {
      const ex = await api.createExport({
        dataset_id: datasetId,
        filter: { ...filter, dataset_id: datasetId },
        val_ratio: valRatio,
        format,
      })
      exports.setData([ex, ...(exports.data ?? []).filter((e) => e.id !== ex.id)])
      exports.reload()
    } catch (e) {
      setCreateError(e)
    } finally {
      setCreating(false)
    }
  }

  const n = matching ?? 0
  const nVal = Math.round(n * valRatio)

  return (
    <section className="card" aria-labelledby="export-heading">
      <div className="card__head">
        <div>
          <h2 id="export-heading">Export train/val split</h2>
          <p>Zip with train.jsonl, val.jsonl and a manifest.</p>
        </div>
      </div>
      <div className="card__body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ fontSize: 13 }}>
          <div className="muted" style={{ fontSize: 12 }}>
            Current filter
          </div>
          <div style={{ overflowWrap: 'anywhere' }}>{describeFilter(filter as Record<string, unknown>)}</div>
        </div>
        <fieldset className="segmented-field">
          <legend>Format</legend>
          <div className="segmented">
            {FORMATS.map((f) => (
              <label key={f.value} className="segmented__opt">
                <input
                  type="radio"
                  name={`${sliderId}-format`}
                  value={f.value}
                  checked={format === f.value}
                  onChange={() => setFormat(f.value)}
                />
                <span>{f.label}</span>
              </label>
            ))}
          </div>
          <span className="field__hint mono" style={{ fontSize: 11.5 }}>
            {FORMATS.find((f) => f.value === format)?.hint}
          </span>
        </fieldset>
        <div className="field">
          <label htmlFor={sliderId}>Validation ratio</label>
          <div className="slider-row">
            <input
              id={sliderId}
              type="range"
              min={0}
              max={0.5}
              step={0.05}
              value={valRatio}
              onChange={(e) => setValRatio(Number(e.target.value))}
              aria-valuetext={`${Math.round(valRatio * 100)} percent validation`}
            />
            <output htmlFor={sliderId}>{Math.round(valRatio * 100)}% val</output>
          </div>
          {matching != null && (
            <span className="field__hint num">
              ≈ {(n - nVal).toLocaleString()} train · {nVal.toLocaleString()} val of {plural(n, 'sample')}
            </span>
          )}
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={create}
          disabled={creating || disabled || matching === 0 || STATIC_DEMO}
        >
          {creating ? <Spinner size={12} /> : <Icon name="package" size={14} />}
          Create export
        </button>
        {STATIC_DEMO && <ReadOnlyNote />}
        {matching === 0 && <span className="field__hint">No samples match the current filter.</span>}
        {createError != null && <ErrorBanner title="Export failed" error={createError} />}
      </div>

      {STATIC_DEMO ? null : exports.initial && exports.loading ? (
        <LoadingBlock label="Loading exports…" />
      ) : exports.error != null && !exports.data ? (
        <div className="card__body" style={{ paddingTop: 0 }}>
          <ErrorBanner title="Could not load exports" error={exports.error} onRetry={exports.reload} />
        </div>
      ) : !exports.data?.length ? (
        <div style={{ borderTop: '1px solid var(--border)' }}>
          <EmptyState icon="package" title="No exports yet">
            Filter to the samples you want — e.g. QC status “pass” — pick a format and validation ratio, then create an
            export.
          </EmptyState>
        </div>
      ) : (
        <ul className="export-list" aria-label="Exports">
          {exports.data.map((ex) => (
            <li key={ex.id} className="export-item">
              <div className="export-item__main">
                <div className="export-item__title">
                  Export #{ex.id}{' '}
                  <span className="muted" style={{ fontWeight: 400 }}>
                    · {ex.format} · {plural(ex.num_samples, 'sample')} · {Math.round(ex.val_ratio * 100)}% val
                  </span>
                </div>
                <div className="export-item__sub" title={formatDate(ex.created_at)}>
                  {describeFilter(ex.filter)} · {relativeTime(ex.created_at)}
                </div>
                {ex.status === 'failed' && (
                  <div className="export-item__sub" style={{ color: 'var(--fail)' }}>
                    {ex.error || 'Export failed'}
                  </div>
                )}
              </div>
              {ex.status === 'ready' ? (
                <a className="btn btn--sm" href={api.exportDownloadUrl(ex.id)} download>
                  <Icon name="download" size={13} /> Download
                </a>
              ) : (
                <StatusPill status={ex.status} />
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
