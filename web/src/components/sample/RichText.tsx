import { useMemo } from 'react'
import type { PiiSpan } from '../../api/types'
import { piiKindClass, piiKindLabel } from '../../lib/pii'
import { codePointToUtf16, segment, splitFences } from '../../lib/text'

/**
 * Field text with whitespace preserved, ``` fences rendered as code blocks and PII spans highlighted.
 * `spans` use code-point offsets (as produced by the API) into `text`.
 */
export function RichText({ text, spans = [] }: { text: string; spans?: PiiSpan[] }) {
  const blocks = useMemo(() => splitFences(text), [text])
  const utf16Spans = useMemo(() => {
    if (!spans.length) return []
    const conv = codePointToUtf16(text)
    return spans.map((s) => ({ ...s, start: conv(s.start), end: conv(s.end) }))
  }, [text, spans])

  if (!text.trim()) return <p className="muted rich-empty">(empty)</p>

  const renderSegments = (start: number, end: number) =>
    segment(start, end, utf16Spans).map((seg, i) =>
      seg.span ? (
        <mark key={i} className={`pii ${piiKindClass(seg.span.kind)}`} title={`PII: ${piiKindLabel(seg.span.kind)}`}>
          {text.slice(seg.start, seg.end)}
          <span className="sr-only"> (PII: {piiKindLabel(seg.span.kind)})</span>
        </mark>
      ) : (
        <span key={i}>{text.slice(seg.start, seg.end)}</span>
      ),
    )

  return (
    <div className="rich">
      {blocks.map((b) =>
        b.kind === 'code' ? (
          <figure key={b.start} className="code-block">
            {b.lang && <figcaption>{b.lang}</figcaption>}
            <pre tabIndex={0}>
              <code>{renderSegments(b.start, b.end)}</code>
            </pre>
          </figure>
        ) : (
          <div key={b.start} className="rich__text">
            {renderSegments(b.start, b.end)}
          </div>
        ),
      )}
    </div>
  )
}
