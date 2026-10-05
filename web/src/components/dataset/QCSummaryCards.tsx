import type { QCCounts } from '../../api/types'
import { Icon, type IconName } from '../Icon'

const CARDS: { key: keyof QCCounts; label: string; icon: IconName }[] = [
  { key: 'pass', label: 'Pass', icon: 'check' },
  { key: 'warn', label: 'Warn', icon: 'warn' },
  { key: 'fail', label: 'Fail', icon: 'x' },
  { key: 'pending', label: 'Pending', icon: 'clock' },
]

export function QCSummaryCards({ counts, loading }: { counts?: QCCounts; loading?: boolean }) {
  const total = counts ? counts.pass + counts.warn + counts.fail + counts.pending : 0
  return (
    <div className="stat-grid" aria-busy={loading}>
      {CARDS.map((c) => {
        const n = counts?.[c.key]
        const pct = counts && total ? Math.round(((n ?? 0) / total) * 100) : null
        return (
          <div key={c.key} className={`stat stat--${c.key}`}>
            <div className="stat__label">
              <Icon name={c.icon} size={13} />
              {c.label}
            </div>
            <div className="stat__value num">{n != null ? n.toLocaleString() : '—'}</div>
            <div className="stat__sub">{pct != null ? `${pct}% of ${total.toLocaleString()} samples` : loading ? 'Loading…' : counts ? 'No samples yet' : 'No data'}</div>
          </div>
        )
      })}
    </div>
  )
}

export function QCDistributionBar({ counts }: { counts: QCCounts }) {
  const total = counts.pass + counts.warn + counts.fail + counts.pending
  if (!total) return null
  const segs = [
    { k: 'pass', v: counts.pass, c: 'var(--pass)' },
    { k: 'warn', v: counts.warn, c: 'var(--warn-mark)' },
    { k: 'fail', v: counts.fail, c: 'var(--fail)' },
    { k: 'pending', v: counts.pending, c: 'var(--border-strong)' },
  ].filter((s) => s.v > 0)
  return (
    <div
      className="dist-bar"
      role="img"
      aria-label={`QC distribution: ${counts.pass} pass, ${counts.warn} warn, ${counts.fail} fail, ${counts.pending} pending`}
    >
      {segs.map((s) => (
        <span key={s.k} style={{ width: `${(s.v / total) * 100}%`, background: s.c }} title={`${s.k}: ${s.v}`} />
      ))}
    </div>
  )
}
