// TypeScript mirrors of api/app/schemas.py — keep in sync with docs/api-contract.md.

export type QCStatus = 'pending' | 'pass' | 'warn' | 'fail'
export type DatasetStatus = 'pending' | 'importing' | 'ready' | 'failed'
export type ExportStatus = 'pending' | 'building' | 'ready' | 'failed'
export type Severity = 'pass' | 'warn' | 'fail'
export type ExportFormat = 'jsonl' | 'chat'

// ---------- datasets ----------
export interface FieldMap {
  prompt?: string | null
  context?: string | null
  response?: string | null
  category?: string | null
  /** chat-format column */
  messages?: string | null
}

export interface ImportRequest {
  hf_repo_id: string
  /** null = the dataset's default config */
  config?: string | null
  split?: string
  max_samples?: number
  /** null = auto-detect */
  fields?: FieldMap | null
}

export interface QCCounts {
  pending: number
  pass: number
  warn: number
  fail: number
}

export interface DatasetOut {
  id: number
  hf_repo_id: string
  config: string
  split: string
  name: string
  revision: string | null
  fields: Record<string, unknown> | null
  num_samples: number
  total_samples: number | null
  /** pending | importing | ready | failed (string on the wire) */
  status: DatasetStatus | (string & {})
  error: string | null
  created_at: string
}

export interface CategoryCount {
  name: string
  count: number
}

export interface HistogramBucket {
  start: number
  end: number
  count: number
}

export interface DatasetDetail extends DatasetOut {
  qc_counts: QCCounts
  categories: CategoryCount[]
  token_histogram: HistogramBucket[]
}

// ---------- samples ----------
export interface SampleOut {
  id: number
  dataset_id: number
  sample_index: number
  category: string | null
  /** first ~160 chars of the prompt */
  prompt_preview: string
  tokens_est: number
  lang: string | null
  qc_status: QCStatus | (string & {})
  qc_score: number | null
}

export interface SampleList {
  items: SampleOut[]
  total: number
}

export interface QCResultOut {
  check_name: string
  passed: boolean
  severity: Severity | (string & {})
  message: string
  details: Record<string, unknown>
}

export interface SampleDetail extends SampleOut {
  prompt: string
  context: string | null
  response: string
  prompt_chars: number
  response_chars: number
  qc_results: QCResultOut[]
  prev_id: number | null
  next_id: number | null
}

// ---------- QC ----------
export interface Enqueued {
  enqueued: number
}

export interface QCSummary {
  counts: QCCounts
  /** check_name -> { warn, fail } */
  by_check: Record<string, { warn?: number; fail?: number }>
}

// ---------- search ----------
export interface SampleFilter {
  dataset_id?: number | null
  qc_status?: QCStatus[] | null
  category?: string | null
  lang?: string | null
  min_tokens?: number | null
  max_tokens?: number | null
  /** case-insensitive match in prompt, context or response */
  text_contains?: string | null
  /** samples with a warn/fail result for this check */
  failed_check?: string | null
}

export interface SearchRequest {
  query: string
  dataset_id?: number | null
  limit?: number
}

export interface SearchResponse {
  filter: SampleFilter
  parser: 'llm' | 'rules'
  items: SampleOut[]
  total: number
}

// ---------- semantic search (embeddings) ----------
export interface EmbeddingStatus {
  model: string
  dim: number
  /** samples of the dataset with a vector for `model` */
  embedded: number
  total: number
}

export interface SemanticSearchRequest {
  query: string
  dataset_id: number
  limit?: number
  /** optional structured filter applied on top; its dataset_id is ignored */
  filter?: SampleFilter
}

export interface SimilarSample extends SampleOut {
  /** cosine similarity, 1 = same direction */
  similarity: number
}

export interface SemanticSearchResponse {
  model: string
  items: SimilarSample[]
}

// ---------- exports ----------
export interface ExportCreate {
  dataset_id: number
  filter: SampleFilter
  val_ratio: number
  format: ExportFormat
}

export interface ExportOut {
  id: number
  dataset_id: number
  filter: Record<string, unknown>
  val_ratio: number
  format: ExportFormat | (string & {})
  status: ExportStatus | (string & {})
  num_samples: number
  error: string | null
  created_at: string
}

// ---------- constants ----------
export const QC_STATUSES: QCStatus[] = ['pass', 'warn', 'fail', 'pending']

export const CHECK_NAMES = [
  'empty_or_short',
  'length_outlier',
  'exact_duplicate',
  'near_duplicate',
  'pii',
  'non_english',
  'refusal_boilerplate',
  'formatting',
] as const

export const CHECK_LABELS: Record<string, string> = {
  empty_or_short: 'Empty / short',
  length_outlier: 'Length outlier',
  exact_duplicate: 'Exact duplicate',
  near_duplicate: 'Near duplicate',
  pii: 'PII',
  non_english: 'Non-English',
  refusal_boilerplate: 'Refusal boilerplate',
  formatting: 'Formatting',
}

export const checkLabel = (name: string) => CHECK_LABELS[name] ?? name.replace(/_/g, ' ')

/** A PII span inside a sample field, from the `pii` check's details.spans. */
export interface PiiSpan {
  field: string
  start: number
  end: number
  kind: string
}
