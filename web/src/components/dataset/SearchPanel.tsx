import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../../api/client'
import type { CategoryCount, QCStatus, SampleFilter } from '../../api/types'
import { CHECK_NAMES, QC_STATUSES, checkLabel } from '../../api/types'
import { cleanFilter, filterChips, type FilterKey } from '../../lib/filter'
import { ErrorBanner } from '../ErrorBanner'
import { Icon } from '../Icon'
import { Spinner } from '../Spinner'

const EXAMPLES = [
  'failed samples with PII',
  'near duplicates',
  'refusals',
  'long answers over 500 tokens',
  'brainstorming samples that need review',
  'non-English samples',
]

export interface SearchMeta {
  query: string
  parser: 'llm' | 'rules'
}

export function SearchPanel({
  datasetId,
  filter,
  categories,
  onFilterChange,
  searchMeta,
  onSearch,
}: {
  datasetId: number
  filter: SampleFilter
  categories: CategoryCount[]
  onFilterChange: (f: SampleFilter, keepSearch?: boolean) => void
  searchMeta: SearchMeta | null
  onSearch: (meta: SearchMeta | null, f: SampleFilter) => void
}) {
  const [query, setQuery] = useState(searchMeta?.query ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  const run = async (q: string) => {
    const text = q.trim()
    if (!text) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.search({ query: text, dataset_id: datasetId, limit: 50 })
      onSearch({ query: text, parser: res.parser }, cleanFilter(res.filter))
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    run(query)
  }

  const removeChip = (key: FilterKey) => {
    const next = { ...filter }
    delete next[key]
    onFilterChange(next, true)
  }

  const chips = filterChips(filter)

  return (
    <section className="card" aria-labelledby="search-heading">
      <div className="card__head">
        <div>
          <h2 id="search-heading">Find samples</h2>
          <p>Describe what you’re looking for in plain English, or set filters manually.</p>
        </div>
      </div>
      <div className="card__body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <form className="search-form" onSubmit={submit} role="search">
          <label htmlFor="nl-search" className="sr-only">
            Natural-language sample search
          </label>
          <div className="search-input-wrap">
            <Icon name="search" size={15} />
            <input
              id="nl-search"
              className="input"
              value={query}
              maxLength={500}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. open_qa samples with emails or phone numbers"
              autoComplete="off"
            />
          </div>
          <button type="submit" className="btn btn--primary" disabled={busy || !query.trim()}>
            {busy ? <Spinner size={12} /> : <Icon name="sparkle" size={13} />}
            Search
          </button>
        </form>

        <div className="examples" role="group" aria-label="Example searches">
          <span>Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              className="example-btn"
              aria-pressed={searchMeta?.query === ex}
              disabled={busy}
              onClick={() => {
                setQuery(ex)
                run(ex)
              }}
            >
              {ex}
            </button>
          ))}
        </div>

        {error != null && <ErrorBanner title="Search failed" error={error} />}

        {(chips.length > 0 || searchMeta) && (
          <div className="chips" aria-live="polite">
            {searchMeta && (
              <span
                className={`pill ${searchMeta.parser === 'llm' ? 'pill--info' : 'pill--outline'}`}
                title={searchMeta.parser === 'llm' ? 'Parsed by an LLM' : 'Parsed by the rule-based parser'}
              >
                {searchMeta.parser === 'llm' ? <Icon name="sparkle" size={11} /> : <Icon name="filter" size={11} />}
                parser: {searchMeta.parser}
              </span>
            )}
            {chips.map((c) => (
              <span key={c.key} className="chip">
                <b>{c.label}</b> {c.value}
                <button type="button" onClick={() => removeChip(c.key)} aria-label={`Remove filter ${c.label} ${c.value}`}>
                  <Icon name="x" size={11} />
                </button>
              </span>
            ))}
            {searchMeta && chips.length === 0 && (
              <span className="muted" style={{ fontSize: 12.5 }}>
                The query didn’t map to any filter — showing all samples.
              </span>
            )}
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => {
                setQuery('')
                onSearch(null, {})
              }}
            >
              Clear all
            </button>
          </div>
        )}

        <ManualFilters filter={filter} categories={categories} onChange={(f) => onFilterChange(f, true)} />
      </div>
    </section>
  )
}

const parseNum = (s: string) => (s.trim() === '' || !Number.isFinite(Number(s)) ? undefined : Math.max(0, Math.round(Number(s))))

function ManualFilters({
  filter,
  categories,
  onChange,
}: {
  filter: SampleFilter
  categories: CategoryCount[]
  onChange: (f: SampleFilter) => void
}) {
  const [minT, setMinT] = useState(filter.min_tokens?.toString() ?? '')
  const [maxT, setMaxT] = useState(filter.max_tokens?.toString() ?? '')
  const [text, setText] = useState(filter.text_contains ?? '')

  // keep inputs in sync when the filter changes elsewhere (search, chip removal, histogram click)
  useEffect(() => setMinT(filter.min_tokens?.toString() ?? ''), [filter.min_tokens])
  useEffect(() => setMaxT(filter.max_tokens?.toString() ?? ''), [filter.max_tokens])
  useEffect(() => setText(filter.text_contains ?? ''), [filter.text_contains])

  // debounce typed edits
  useEffect(() => {
    const t = setTimeout(() => {
      const a = parseNum(minT)
      const b = parseNum(maxT)
      const c = text.trim() || undefined
      if (
        a !== (filter.min_tokens ?? undefined) ||
        b !== (filter.max_tokens ?? undefined) ||
        c !== (filter.text_contains ?? undefined)
      ) {
        onChange(cleanFilter({ ...filter, min_tokens: a, max_tokens: b, text_contains: c }))
      }
    }, 450)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minT, maxT, text])

  const toggleStatus = (s: QCStatus) => {
    const cur = new Set(filter.qc_status ?? [])
    if (cur.has(s)) cur.delete(s)
    else cur.add(s)
    onChange(cleanFilter({ ...filter, qc_status: QC_STATUSES.filter((x) => cur.has(x)) }))
  }

  const rangeInvalid = minT !== '' && maxT !== '' && Number(minT) > Number(maxT)
  const categoryOptions =
    filter.category && !categories.some((c) => c.name === filter.category)
      ? [...categories, { name: filter.category, count: 0 }]
      : categories

  return (
    <div className="manual-filters">
      <fieldset className="checkbox-group">
        <legend>QC status</legend>
        {QC_STATUSES.map((s) => (
          <label key={s} className="toggle-chip">
            <input type="checkbox" checked={filter.qc_status?.includes(s) ?? false} onChange={() => toggleStatus(s)} />
            <i className="swatch" style={{ background: `var(--${s === 'warn' ? 'warn-mark' : s})`, borderRadius: '50%' }} />
            {s}
          </label>
        ))}
      </fieldset>
      <div className="form-row" style={{ alignItems: 'flex-start' }}>
        <div className="field" style={{ flex: '1 1 160px' }}>
          <label htmlFor="f-category">Category</label>
          <select
            id="f-category"
            className="input"
            value={filter.category ?? ''}
            onChange={(e) => onChange(cleanFilter({ ...filter, category: e.target.value || undefined }))}
          >
            <option value="">Any</option>
            {categoryOptions.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.count.toLocaleString()})
              </option>
            ))}
          </select>
        </div>
        <div className="field" style={{ flex: '1 1 160px' }}>
          <label htmlFor="f-check">Flagged by check</label>
          <select
            id="f-check"
            className="input"
            value={filter.failed_check ?? ''}
            onChange={(e) => onChange(cleanFilter({ ...filter, failed_check: e.target.value || undefined }))}
          >
            <option value="">Any</option>
            {CHECK_NAMES.map((c) => (
              <option key={c} value={c}>
                {checkLabel(c)}
              </option>
            ))}
          </select>
        </div>
        <div className="field field--narrow">
          <label htmlFor="f-min">Min tokens</label>
          <input
            id="f-min"
            className="input num"
            type="number"
            min={0}
            step={1}
            inputMode="numeric"
            value={minT}
            onChange={(e) => setMinT(e.target.value)}
            aria-invalid={rangeInvalid}
          />
        </div>
        <div className="field field--narrow">
          <label htmlFor="f-max">Max tokens</label>
          <input
            id="f-max"
            className="input num"
            type="number"
            min={0}
            step={1}
            inputMode="numeric"
            value={maxT}
            onChange={(e) => setMaxT(e.target.value)}
            aria-invalid={rangeInvalid}
            aria-describedby={rangeInvalid ? 'tok-err' : undefined}
          />
        </div>
      </div>
      <div className="field">
        <label htmlFor="f-text">Text contains</label>
        <input
          id="f-text"
          className="input"
          type="search"
          value={text}
          maxLength={200}
          placeholder="Matches prompt, context or response (case-insensitive)"
          onChange={(e) => setText(e.target.value)}
        />
      </div>
      {rangeInvalid && (
        <span id="tok-err" className="field__error">
          Min tokens is greater than max tokens — no samples will match.
        </span>
      )}
    </div>
  )
}
