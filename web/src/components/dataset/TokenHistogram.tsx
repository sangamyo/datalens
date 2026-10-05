import { useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import type { HistogramBucket } from '../../api/types'
import { EmptyState } from '../EmptyState'

const H = 180
const PAD = { top: 10, right: 8, bottom: 26, left: 40 }

function niceTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0]
  const raw = max / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw
  const ticks: number[] = []
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(Math.round(v))
  if (ticks[ticks.length - 1] < max) ticks.push(Math.round(ticks[ticks.length - 1] + step))
  return ticks
}

const compact = (n: number) => (n >= 10000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n))

/** Rounded-top bar anchored to the baseline. */
function barPath(x: number, y: number, w: number, h: number, r: number) {
  if (h <= 0) return ''
  const rr = Math.min(r, w / 2, h)
  return `M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z`
}

/** Token-length histogram (hand-drawn SVG). Hover/focus a bar for its count; click to filter by that range. */
export function TokenHistogram({
  buckets,
  activeRange,
  onSelect,
}: {
  buckets: HistogramBucket[]
  activeRange?: { min?: number | null; max?: number | null }
  /** Inclusive token range of the clicked bucket, or null to clear. */
  onSelect?: (range: { min: number; max: number } | null) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)
  const [hover, setHover] = useState<number | null>(null)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const update = () => setWidth(Math.max(240, el.clientWidth))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const total = buckets.reduce((s, b) => s + b.count, 0)
  if (!buckets.length || total === 0) {
    return (
      <div className="hist" ref={wrapRef}>
        <EmptyState icon="chart" title="No token data yet">
          The histogram appears once samples are imported.
        </EmptyState>
      </div>
    )
  }

  const maxCount = Math.max(...buckets.map((b) => b.count), 1)
  const ticks = niceTicks(maxCount)
  const yMax = ticks[ticks.length - 1] || 1
  const innerW = width - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom
  const slot = innerW / buckets.length
  const gap = Math.min(4, Math.max(2, slot * 0.12))
  const barW = Math.max(1, slot - gap)
  const y = (v: number) => PAD.top + innerH - (v / yMax) * innerH
  const labelEvery = Math.ceil(buckets.length / Math.max(2, Math.floor(innerW / 48)))

  // Buckets are [start, end) except the last, which is open-ended up to the max value; the
  // filter's max_tokens is inclusive, so a bucket maps to start..end-1 (start..end for the last).
  const rangeOf = (i: number) => ({
    min: buckets[i].start,
    max: i === buckets.length - 1 ? buckets[i].end : Math.max(buckets[i].start, buckets[i].end - 1),
  })
  const isActive = (i: number) => {
    const r = rangeOf(i)
    return activeRange != null && activeRange.min === r.min && activeRange.max === r.max
  }
  const hasActive = buckets.some((_, i) => isActive(i))

  const onKey = (e: KeyboardEvent<SVGGElement>, i: number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onSelect?.(isActive(i) ? null : rangeOf(i))
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      const next = Math.max(0, Math.min(buckets.length - 1, i + (e.key === 'ArrowRight' ? 1 : -1)))
      wrapRef.current?.querySelectorAll<SVGGElement>('g.hist-bar')[next]?.focus()
    }
  }

  const hb = hover != null ? buckets[hover] : null
  const tipX = hover != null ? PAD.left + hover * slot + slot / 2 : 0

  return (
    <div className="hist" ref={wrapRef}>
      <svg
        width={width}
        height={H}
        className="hist-svg"
        role="group"
        aria-label={`Token length histogram, ${buckets.length} buckets, ${total.toLocaleString()} samples`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} />
            <text x={PAD.left - 6} y={y(t)} dy="0.32em" textAnchor="end">
              {compact(t)}
            </text>
          </g>
        ))}
        {buckets.map((b, i) => {
          const x = PAD.left + i * slot + gap / 2
          const top = y(b.count)
          const h = PAD.top + innerH - top
          const active = isActive(i)
          return (
            <g
              key={`${b.start}-${b.end}`}
              className={`hist-bar${active ? ' is-active' : ''}${hasActive && !active ? ' is-dim' : ''}`}
              tabIndex={0}
              role="button"
              aria-pressed={active}
              aria-label={`${b.start.toLocaleString()} to ${b.end.toLocaleString()} tokens: ${b.count.toLocaleString()} samples. ${active ? 'Clear filter' : 'Filter to this range'}`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover((h) => (h === i ? null : h))}
              onFocus={() => setHover(i)}
              onBlur={() => setHover((h) => (h === i ? null : h))}
              onClick={() => onSelect?.(active ? null : rangeOf(i))}
              onKeyDown={(e) => onKey(e, i)}
            >
              {/* full-height hit target, larger than the mark */}
              <rect className="hist-hit" x={PAD.left + i * slot} y={PAD.top} width={slot} height={innerH} />
              <path className="hist-mark" d={barPath(x, top, barW, h, 3)} />
              {b.count > 0 && h < 1 && <rect className="hist-mark" x={x} y={PAD.top + innerH - 1} width={barW} height={1} />}
            </g>
          )
        })}
        <line className="axis-line" x1={PAD.left} x2={width - PAD.right} y1={PAD.top + innerH} y2={PAD.top + innerH} />
        {buckets.map((b, i) =>
          i % labelEvery === 0 ? (
            <text key={b.start} x={PAD.left + i * slot} y={H - 8} textAnchor={i === 0 ? 'start' : 'middle'}>
              {compact(b.start)}
            </text>
          ) : null,
        )}
        <text x={width - PAD.right} y={H - 8} textAnchor="end">
          {compact(buckets[buckets.length - 1].end)}
        </text>
      </svg>
      {hb && (
        <div
          className="chart-tip"
          role="presentation"
          style={{
            left: Math.min(Math.max(tipX, 70), width - 70),
            top: Math.max(0, y(hb.count) - 8),
          }}
        >
          <div className="chart-tip__title num">
            {hb.start.toLocaleString()}–{hb.end.toLocaleString()} tokens
          </div>
          <div className="num">
            <b>{hb.count.toLocaleString()}</b> samples <span className="muted">· {Math.round((hb.count / total) * 100)}%</span>
          </div>
        </div>
      )}
      <div className="legend-inline" style={{ marginTop: 6 }}>
        <span>Estimated tokens per sample (prompt + context + response) · click a bar to filter</span>
      </div>
    </div>
  )
}
