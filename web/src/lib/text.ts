import type { PiiSpan, QCResultOut } from '../api/types'

/** A slice of a field's text: plain prose or the inside of a ``` fence. Offsets are UTF-16 indices. */
export interface Block {
  kind: 'text' | 'code'
  lang?: string
  start: number
  end: number
}

const FENCE_RE = /```([^\n`]*)\n([\s\S]*?)(?:\n?```|$)/g

/** Split text into prose and fenced code blocks, keeping offsets into the original string. */
export function splitFences(text: string): Block[] {
  const blocks: Block[] = []
  let last = 0
  FENCE_RE.lastIndex = 0
  for (let m = FENCE_RE.exec(text); m; m = FENCE_RE.exec(text)) {
    if (m.index > last) blocks.push({ kind: 'text', start: last, end: m.index })
    const start = m.index + 3 + m[1].length + 1
    blocks.push({ kind: 'code', lang: m[1].trim() || undefined, start, end: start + m[2].length })
    last = m.index + m[0].length
    if (m[0].length === 0) break
  }
  if (last < text.length) blocks.push({ kind: 'text', start: last, end: text.length })
  // trim the newlines that separate prose from fences, and drop prose blocks that are only whitespace
  for (const b of blocks) {
    if (b.kind !== 'text') continue
    while (b.start < b.end && (text[b.start] === '\n' || text[b.start] === '\r')) b.start++
    while (b.end > b.start && /\s/.test(text[b.end - 1])) b.end--
  }
  return blocks.filter((b) => b.kind === 'code' || text.slice(b.start, b.end).trim() !== '')
}

/** Python (the API) counts code points; JS strings count UTF-16 units. Convert span offsets. */
export function codePointToUtf16(text: string): (cp: number) => number {
  // fast path: no astral characters
  if (!/[\uD800-\uDBFF]/.test(text)) return (cp) => Math.min(cp, text.length)
  const map: number[] = []
  let u = 0
  for (const ch of text) {
    map.push(u)
    u += ch.length
  }
  map.push(u)
  return (cp) => map[Math.min(Math.max(cp, 0), map.length - 1)]
}

export interface Segment {
  start: number
  end: number
  span?: PiiSpan
}

/** Cut [start, end) into plain and highlighted segments. `spans` must already be UTF-16 offsets. */
export function segment(start: number, end: number, spans: PiiSpan[]): Segment[] {
  const out: Segment[] = []
  let pos = start
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    const a = Math.max(s.start, pos)
    const b = Math.min(s.end, end)
    if (b <= a) continue
    if (a > pos) out.push({ start: pos, end: a })
    out.push({ start: a, end: b, span: s })
    pos = b
  }
  if (pos < end) out.push({ start: pos, end })
  return out
}

export function piiSpans(results: QCResultOut[] | undefined): PiiSpan[] {
  const r = results?.find((x) => x.check_name === 'pii')
  const spans = r?.details?.spans
  if (!Array.isArray(spans)) return []
  return spans.filter(
    (s): s is PiiSpan =>
      !!s &&
      typeof s === 'object' &&
      typeof (s as PiiSpan).field === 'string' &&
      typeof (s as PiiSpan).start === 'number' &&
      typeof (s as PiiSpan).end === 'number',
  )
}

export interface RelatedSample {
  id: number
  jaccard?: number
}

/** Normalise duplicate lists: plain ids (`[12, 40]`) or `{id, jaccard}` objects. */
export function relatedSamples(v: unknown): RelatedSample[] {
  if (!Array.isArray(v)) return []
  const out: RelatedSample[] = []
  for (const x of v) {
    if (typeof x === 'number') out.push({ id: x })
    else if (x && typeof x === 'object' && typeof (x as { id?: unknown }).id === 'number') {
      const j = (x as { jaccard?: unknown; similarity?: unknown }).jaccard ?? (x as { similarity?: unknown }).similarity
      out.push({ id: (x as { id: number }).id, jaccard: typeof j === 'number' ? j : undefined })
    } else if (Array.isArray(x) && typeof x[0] === 'number') {
      out.push({ id: x[0], jaccard: typeof x[1] === 'number' ? x[1] : undefined })
    }
  }
  return out
}
