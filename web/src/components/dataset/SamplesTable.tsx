import { useRef, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import type { SampleList } from '../../api/types'
import { formatScore } from '../../lib/format'
import { EmptyState } from '../EmptyState'
import { ErrorBanner } from '../ErrorBanner'
import { Icon } from '../Icon'
import { QCPill } from '../Pills'
import { LoadingBlock, Spinner } from '../Spinner'

export function SamplesTable({
  data,
  error,
  loading,
  initial,
  offset,
  pageSize,
  onPage,
  onRetry,
  filtered,
  onClearFilter,
  importing,
}: {
  data?: SampleList
  error: unknown
  loading: boolean
  initial: boolean
  offset: number
  pageSize: number
  onPage: (offset: number) => void
  onRetry: () => void
  filtered: boolean
  onClearFilter: () => void
  importing?: boolean
}) {
  const navigate = useNavigate()
  const bodyRef = useRef<HTMLTableSectionElement>(null)

  const onRowKey = (e: KeyboardEvent<HTMLTableRowElement>, id: number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      navigate(`/samples/${id}`)
      return
    }
    const rows = Array.from(bodyRef.current?.querySelectorAll<HTMLTableRowElement>('tr.row-link') ?? [])
    const i = rows.indexOf(e.currentTarget)
    let target: HTMLTableRowElement | undefined
    if (e.key === 'ArrowDown') target = rows[Math.min(rows.length - 1, i + 1)]
    else if (e.key === 'ArrowUp') target = rows[Math.max(0, i - 1)]
    else if (e.key === 'Home') target = rows[0]
    else if (e.key === 'End') target = rows[rows.length - 1]
    if (target) {
      e.preventDefault()
      target.focus()
    }
  }

  const total = data?.total ?? 0
  const from = total ? offset + 1 : 0
  const to = Math.min(offset + pageSize, total)

  return (
    <section className="card" aria-labelledby="samples-heading">
      <div className="card__head">
        <div>
          <h2 id="samples-heading">Samples</h2>
          <p>
            {data ? `${total.toLocaleString()} ${filtered ? 'matching' : 'total'}` : ' '}
            {loading && !initial && (
              <>
                {' '}
                <Spinner size={10} label="Updating" />
              </>
            )}
          </p>
        </div>
        <span className="muted hide-xs" style={{ fontSize: 12 }}>
          <kbd>↑</kbd> <kbd>↓</kbd> to move · <kbd>Enter</kbd> to open
        </span>
      </div>

      {initial && loading ? (
        <LoadingBlock label="Loading samples…" />
      ) : error != null && !data ? (
        <div className="card__body">
          <ErrorBanner title="Could not load samples" error={error} onRetry={onRetry} />
        </div>
      ) : !data || data.items.length === 0 ? (
        filtered ? (
          <EmptyState
            icon="filter"
            title="No samples match"
            actions={
              <button type="button" className="btn" onClick={onClearFilter}>
                Clear filters
              </button>
            }
          >
            Try loosening the filters or a different search.
          </EmptyState>
        ) : (
          <EmptyState icon="database" title="No samples yet">
            {importing ? 'Samples appear here as the import progresses.' : 'This dataset has no samples.'}
          </EmptyState>
        )
      ) : (
        <>
          {error != null && (
            <div className="card__body" style={{ paddingBottom: 0 }}>
              <ErrorBanner title="Could not refresh samples" error={error} onRetry={onRetry} />
            </div>
          )}
          <div className={`table-wrap${loading ? ' table-stale' : ''}`}>
            <table className="table table--samples">
              <caption className="sr-only">Samples. Use arrow keys to move between rows and Enter to open one.</caption>
              <thead>
                <tr>
                  <th scope="col" className="r">
                    #
                  </th>
                  <th scope="col" className="hide-sm">
                    Category
                  </th>
                  <th scope="col">Prompt</th>
                  <th scope="col" className="r">
                    Tokens
                  </th>
                  <th scope="col" className="hide-xs">
                    Lang
                  </th>
                  <th scope="col">QC</th>
                  <th scope="col" className="r hide-xs">
                    Score
                  </th>
                </tr>
              </thead>
              <tbody ref={bodyRef}>
                {data.items.map((s) => (
                  <tr
                    key={s.id}
                    className="row-link"
                    tabIndex={0}
                    onClick={() => navigate(`/samples/${s.id}`)}
                    onKeyDown={(e) => onRowKey(e, s.id)}
                    aria-label={`Sample ${s.sample_index}, ${s.qc_status}: ${s.prompt_preview.slice(0, 80)}`}
                  >
                    <td className="r num mono muted">{s.sample_index}</td>
                    <td className="hide-sm">
                      {s.category ? <span className="tag">{s.category}</span> : <span className="muted">—</span>}
                    </td>
                    <td className="prompt-cell">
                      <span className="prompt-cell__text" title={s.prompt_preview}>
                        {s.prompt_preview || <span className="muted">(empty prompt)</span>}
                      </span>
                      {s.category && <span className="cell-sub show-sm">{s.category}</span>}
                    </td>
                    <td className="r num">{s.tokens_est.toLocaleString()}</td>
                    <td className="hide-xs mono secondary" style={{ fontSize: 12 }}>
                      {s.lang ?? '—'}
                    </td>
                    <td>
                      <QCPill status={s.qc_status} />
                    </td>
                    <td className="r num hide-xs">{formatScore(s.qc_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {total > pageSize && (
            <nav className="pager" aria-label="Sample pages">
              <span className="num">
                {from.toLocaleString()}–{to.toLocaleString()} of {total.toLocaleString()}
              </span>
              <div className="pager__buttons">
                <button
                  type="button"
                  className="btn btn--sm"
                  disabled={offset === 0}
                  onClick={() => onPage(Math.max(0, offset - pageSize))}
                >
                  <Icon name="chevronLeft" size={13} /> Prev
                </button>
                <button
                  type="button"
                  className="btn btn--sm"
                  disabled={to >= total}
                  onClick={() => onPage(offset + pageSize)}
                >
                  Next <Icon name="chevronRight" size={13} />
                </button>
              </div>
            </nav>
          )}
        </>
      )}
    </section>
  )
}
