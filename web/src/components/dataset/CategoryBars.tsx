import type { CategoryCount } from '../../api/types'
import { EmptyState } from '../EmptyState'

const MAX_ROWS = 14

/** Horizontal bars: samples per category, largest first. Click a row to filter. */
export function CategoryBars({
  categories,
  active,
  onSelect,
}: {
  categories: CategoryCount[]
  active?: string | null
  onSelect: (name: string | null) => void
}) {
  if (!categories.length) {
    return (
      <EmptyState icon="tag" title="No categories">
        This dataset has no category column (or none was mapped).
      </EmptyState>
    )
  }
  const sorted = [...categories].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
  const shown = sorted.slice(0, MAX_ROWS)
  const rest = sorted.slice(MAX_ROWS)
  const total = sorted.reduce((s, c) => s + c.count, 0)
  const max = Math.max(...shown.map((c) => c.count), 1)
  return (
    <>
      <ul className="cat-bars" aria-label="Samples per category">
        {shown.map((c) => {
          const isActive = active === c.name
          const pct = total ? Math.round((c.count / total) * 100) : 0
          return (
            <li key={c.name}>
              <button
                type="button"
                className="cat-bar"
                aria-pressed={isActive}
                onClick={() => onSelect(isActive ? null : c.name)}
                aria-label={`${c.name}: ${c.count} samples (${pct}%). ${isActive ? 'Clear filter' : 'Filter samples'}`}
                title={isActive ? 'Clear this filter' : `Show ${c.name} samples`}
              >
                <span className="cat-bar__name">{c.name}</span>
                <span className="cat-bar__track" aria-hidden="true">
                  <span style={{ width: `${(c.count / max) * 100}%` }} />
                </span>
                <span className="cat-bar__count num">
                  {c.count.toLocaleString()} <span className="muted">{pct}%</span>
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      {rest.length > 0 && (
        <p className="field__hint" style={{ margin: '10px 0 0' }}>
          + {rest.length} more categories ({rest.reduce((s, c) => s + c.count, 0).toLocaleString()} samples) — use the
          category filter.
        </p>
      )}
    </>
  )
}
