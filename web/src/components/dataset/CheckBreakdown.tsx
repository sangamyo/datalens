import type { QCSummary } from '../../api/types'
import { CHECK_NAMES, checkLabel } from '../../api/types'
import { EmptyState } from '../EmptyState'

/** Horizontal bars: how many samples each check flagged (warn + fail), most-failing first. */
export function CheckBreakdown({
  summary,
  totalSamples,
  activeCheck,
  onSelect,
}: {
  summary: QCSummary
  totalSamples: number
  activeCheck?: string | null
  onSelect: (check: string | null) => void
}) {
  const names = Array.from(new Set([...CHECK_NAMES, ...Object.keys(summary.by_check)]))
  const rows = names
    .map((name) => {
      const c = summary.by_check[name] ?? {}
      return { name, warn: c.warn ?? 0, fail: c.fail ?? 0 }
    })
    .sort((a, b) => b.fail - a.fail || b.warn - a.warn || a.name.localeCompare(b.name))
  const scale = Math.max(totalSamples, ...rows.map((r) => r.warn + r.fail), 1)
  const checked = summary.counts.pass + summary.counts.warn + summary.counts.fail

  if (checked === 0) {
    return (
      <EmptyState icon="chart" title="No QC results yet">
        Results appear here as quality checks finish for each sample.
      </EmptyState>
    )
  }

  return (
    <>
      <ul className="check-bars" aria-label="Samples flagged per check">
        {rows.map((r) => {
          const active = activeCheck === r.name
          return (
            <li key={r.name}>
              <button
                type="button"
                className="check-bar"
                aria-pressed={active}
                onClick={() => onSelect(active ? null : r.name)}
                title={active ? 'Clear this filter' : `Show samples flagged by ${checkLabel(r.name)}`}
                aria-label={`${checkLabel(r.name)}: ${r.fail} fail, ${r.warn} warn. ${active ? 'Clear filter' : 'Filter samples'}`}
              >
                <span className="check-bar__name">{checkLabel(r.name)}</span>
                <span className="check-bar__track" aria-hidden="true">
                  {r.fail > 0 && <span style={{ width: `${(r.fail / scale) * 100}%`, background: 'var(--fail)' }} />}
                  {r.warn > 0 && <span style={{ width: `${(r.warn / scale) * 100}%`, background: 'var(--warn-mark)' }} />}
                </span>
                <span className="check-bar__count">
                  {r.fail + r.warn === 0 ? (
                    <span className="muted">0</span>
                  ) : (
                    <>
                      <span style={{ color: r.fail ? 'var(--fail)' : 'var(--text-muted)' }}>{r.fail}</span>
                      <span className="muted"> / </span>
                      <span style={{ color: r.warn ? 'var(--warn)' : 'var(--text-muted)' }}>{r.warn}</span>
                    </>
                  )}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <div className="legend-inline" style={{ marginTop: 14 }}>
        <span>
          <i className="swatch" style={{ background: 'var(--fail)' }} /> fail
        </span>
        <span>
          <i className="swatch" style={{ background: 'var(--warn-mark)' }} /> warn
        </span>
        <span>Counts: fail / warn · click a check to filter</span>
      </div>
    </>
  )
}
