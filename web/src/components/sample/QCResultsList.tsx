import { Link } from 'react-router-dom'
import type { QCResultOut } from '../../api/types'
import { checkLabel } from '../../api/types'
import { formatNumber } from '../../lib/format'
import { piiKindLabel } from '../../lib/pii'
import { relatedSamples } from '../../lib/text'
import { EmptyState } from '../EmptyState'
import { QCPill } from '../Pills'

const ORDER: Record<string, number> = { fail: 0, warn: 1, pass: 2 }
const RELATED_KEYS: Record<string, string> = {
  duplicate_of: 'Duplicate of',
  similar_to: 'Similar to',
  same_prompt_as: 'Same prompt as',
}
const HIDDEN_KEYS = new Set(['spans', 'kinds', ...Object.keys(RELATED_KEYS)])

function formatDetail(v: unknown): string {
  if (typeof v === 'number') return formatNumber(v, 3)
  if (Array.isArray(v)) return `[${v.slice(0, 6).map(formatDetail).join(', ')}${v.length > 6 ? ', …' : ''}]`
  if (v && typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

export function QCResultsList({ results, pending }: { results: QCResultOut[]; pending: boolean }) {
  if (!results.length) {
    return (
      <EmptyState icon="clock" title={pending ? 'QC pending' : 'No QC results'}>
        {pending ? 'Quality checks for this sample haven’t finished yet.' : 'No quality checks have been run on this sample.'}
      </EmptyState>
    )
  }
  const sorted = [...results].sort(
    (a, b) => (ORDER[a.severity] ?? 3) - (ORDER[b.severity] ?? 3) || a.check_name.localeCompare(b.check_name),
  )
  return (
    <ul className="qc-list">
      {sorted.map((r) => {
        const details = r.details ?? {}
        const plain = Object.entries(details).filter(([k]) => !HIDDEN_KEYS.has(k))
        const kinds =
          details.kinds && typeof details.kinds === 'object' ? Object.entries(details.kinds as Record<string, number>) : []
        const related = Object.keys(RELATED_KEYS)
          .map((k) => ({ key: k, items: relatedSamples(details[k]) }))
          .filter((x) => x.items.length > 0)
        return (
          <li key={r.check_name} className="qc-item">
            <QCPill status={r.severity} />
            <span className="qc-item__name">
              {checkLabel(r.check_name)} <code className="muted">{r.check_name}</code>
            </span>
            <span className="qc-item__msg">{r.message}</span>
            {kinds.length > 0 && (
              <span className="qc-item__details">
                {kinds.map(([k, n]) => `${piiKindLabel(k)} × ${n}`).join('  ·  ')}
              </span>
            )}
            {related.map(({ key, items }) => (
              <span key={key} className="qc-item__related">
                <span className="muted">{RELATED_KEYS[key]}:</span>
                {items.map((it) => (
                  <Link key={it.id} to={`/samples/${it.id}`} className="related-link" title={`Open sample id ${it.id}`}>
                    id {it.id}
                    {it.jaccard != null && <span className="related-link__score">J {it.jaccard.toFixed(2)}</span>}
                  </Link>
                ))}
              </span>
            ))}
            {plain.length > 0 && (
              <span className="qc-item__details">
                {plain
                  .slice(0, 6)
                  .map(([k, v]) => `${k}=${formatDetail(v)}`)
                  .join('  ·  ')}
              </span>
            )}
          </li>
        )
      })}
    </ul>
  )
}
