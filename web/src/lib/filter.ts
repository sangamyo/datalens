import type { SampleFilter, QCStatus } from '../api/types'
import { QC_STATUSES, checkLabel } from '../api/types'

export type FilterKey = Exclude<keyof SampleFilter, 'dataset_id'>

const NUM_KEYS = ['min_tokens', 'max_tokens'] as const
const STR_KEYS = ['category', 'lang', 'text_contains', 'failed_check'] as const

/** Read a SampleFilter out of URL search params. */
export function filterFromParams(p: URLSearchParams): SampleFilter {
  const f: SampleFilter = {}
  const statuses = p.getAll('qc_status').filter((s): s is QCStatus => (QC_STATUSES as string[]).includes(s))
  if (statuses.length) f.qc_status = statuses
  for (const k of NUM_KEYS) {
    const v = p.get(k)
    if (v != null && v !== '' && Number.isFinite(Number(v))) f[k] = Number(v)
  }
  for (const k of STR_KEYS) {
    const v = p.get(k)
    if (v) f[k] = v
  }
  return f
}

/** Strip null/empty fields and dataset_id. */
export function cleanFilter(f: SampleFilter): SampleFilter {
  const out: SampleFilter = {}
  if (f.qc_status && f.qc_status.length) out.qc_status = QC_STATUSES.filter((s) => f.qc_status!.includes(s))
  for (const k of NUM_KEYS) {
    const v = f[k]
    if (v != null && Number.isFinite(v)) out[k] = v
  }
  for (const k of STR_KEYS) if (f[k]) out[k] = f[k]
  return out
}

export function isEmptyFilter(f: SampleFilter): boolean {
  return Object.keys(cleanFilter(f)).length === 0
}

export interface FilterChip {
  key: FilterKey
  label: string
  value: string
}

export function filterChips(f: SampleFilter): FilterChip[] {
  const chips: FilterChip[] = []
  if (f.qc_status?.length) chips.push({ key: 'qc_status', label: 'QC', value: f.qc_status.join(' or ') })
  if (f.failed_check) chips.push({ key: 'failed_check', label: 'Flagged by', value: checkLabel(f.failed_check) })
  if (f.category) chips.push({ key: 'category', label: 'Category', value: f.category })
  if (f.lang) chips.push({ key: 'lang', label: 'Language', value: f.lang })
  if (f.min_tokens != null) chips.push({ key: 'min_tokens', label: 'Tokens ≥', value: f.min_tokens.toLocaleString() })
  if (f.max_tokens != null) chips.push({ key: 'max_tokens', label: 'Tokens ≤', value: f.max_tokens.toLocaleString() })
  if (f.text_contains) chips.push({ key: 'text_contains', label: 'Contains', value: `“${f.text_contains}”` })
  return chips
}

export function describeFilter(f: Record<string, unknown>): string {
  const chips = filterChips(cleanFilter(f as SampleFilter))
  if (!chips.length) return 'All samples'
  return chips.map((c) => `${c.label} ${c.value}`).join(' · ')
}

// The dataset page remembers its query string so the sample viewer's breadcrumb can go back to it.
const key = (datasetId: number) => `datalens:dataset-search:${datasetId}`

export function rememberDatasetSearch(datasetId: number, search: string) {
  try {
    sessionStorage.setItem(key(datasetId), search)
  } catch {
    // storage unavailable — breadcrumb falls back to the unfiltered page
  }
}

export function recallDatasetSearch(datasetId: number): string {
  try {
    return sessionStorage.getItem(key(datasetId)) ?? ''
  } catch {
    return ''
  }
}
