import type {
  DatasetDetail,
  DatasetOut,
  EmbeddingStatus,
  Enqueued,
  ExportCreate,
  ExportOut,
  ImportRequest,
  QCResultOut,
  QCSummary,
  SampleDetail,
  SampleFilter,
  SampleList,
  SearchRequest,
  SearchResponse,
  SemanticSearchRequest,
  SemanticSearchResponse,
  SimilarSample,
} from './types'

export const API_BASE = '/api'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function detailToMessage(detail: unknown): string | null {
  if (typeof detail === 'string') return detail
  // FastAPI 422s: [{loc, msg, type}, ...]
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => {
        if (d && typeof d === 'object' && 'msg' in d) {
          const loc = Array.isArray((d as { loc?: unknown[] }).loc)
            ? (d as { loc: unknown[] }).loc.filter((p) => p !== 'body').join('.')
            : ''
          return loc ? `${loc}: ${String((d as { msg: unknown }).msg)}` : String((d as { msg: unknown }).msg)
        }
        return null
      })
      .filter(Boolean)
    if (msgs.length) return msgs.join('; ')
  }
  return null
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init.body ? { 'Content-Type': 'application/json' } : {}),
        ...init.headers,
      },
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the DataLens API. Is the backend running on port 8000?')
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText || 'Request failed'}`
    const text = await res.text().catch(() => '')
    if (text) {
      try {
        const body = JSON.parse(text) as { detail?: unknown }
        message = detailToMessage(body.detail) ?? message
      } catch {
        // Vite's proxy returns a non-JSON 500/502 when the backend is down.
        if (res.status >= 500 && res.status !== 501) {
          message = `API unavailable (${res.status}). Is the backend running on port 8000?`
        }
      }
    }
    throw new ApiError(res.status, message)
  }
  if (res.status === 204) return undefined as T
  const text = await res.text()
  return (text ? JSON.parse(text) : undefined) as T
}

const json = (body: unknown) => JSON.stringify(body)

const SCALAR_KEYS = ['category', 'lang', 'min_tokens', 'max_tokens', 'text_contains', 'failed_check'] as const

/** Build a querystring from a SampleFilter (qc_status is repeatable). */
export function filterToParams(filter: SampleFilter, extra: Record<string, number> = {}): string {
  const p = new URLSearchParams()
  filter.qc_status?.forEach((s) => p.append('qc_status', s))
  for (const k of SCALAR_KEYS) {
    const v = filter[k]
    if (v !== null && v !== undefined && v !== '') p.set(k, String(v))
  }
  for (const [k, v] of Object.entries(extra)) p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

export const api = {
  // datasets
  listDatasets: () => request<DatasetOut[]>('/datasets'),
  getDataset: (id: number) => request<DatasetDetail>(`/datasets/${id}`),
  importDataset: (body: ImportRequest) =>
    request<DatasetOut>('/datasets/import', { method: 'POST', body: json(body) }),
  deleteDataset: (id: number) => request<void>(`/datasets/${id}`, { method: 'DELETE' }),

  // samples
  listSamples: (datasetId: number, filter: SampleFilter, limit = 50, offset = 0) =>
    request<SampleList>(`/datasets/${datasetId}/samples${filterToParams(filter, { limit, offset })}`),
  getSample: (id: number) => request<SampleDetail>(`/samples/${id}`),

  // QC
  runDatasetQC: (id: number) => request<Enqueued>(`/datasets/${id}/qc`, { method: 'POST' }),
  runSampleQC: (id: number) => request<Enqueued>(`/samples/${id}/qc`, { method: 'POST' }),
  getSampleQC: (id: number) => request<QCResultOut[]>(`/samples/${id}/qc`),
  getQCSummary: (id: number) => request<QCSummary>(`/datasets/${id}/qc/summary`),

  // search
  search: (body: SearchRequest) =>
    request<SearchResponse>('/search', { method: 'POST', body: json(body) }),

  // semantic search (embeddings)
  getEmbeddingStatus: (datasetId: number) => request<EmbeddingStatus>(`/datasets/${datasetId}/embeddings`),
  embedDataset: (datasetId: number) => request<Enqueued>(`/datasets/${datasetId}/embeddings`, { method: 'POST' }),
  semanticSearch: (body: SemanticSearchRequest) =>
    request<SemanticSearchResponse>('/search/semantic', { method: 'POST', body: json(body) }),
  similarSamples: (sampleId: number, limit = 8) => request<SimilarSample[]>(`/samples/${sampleId}/similar?limit=${limit}`),

  // exports
  createExport: (body: ExportCreate) =>
    request<ExportOut>('/exports', { method: 'POST', body: json(body) }),
  listExports: (datasetId: number) => request<ExportOut[]>(`/exports?dataset_id=${datasetId}`),
  getExport: (id: number) => request<ExportOut>(`/exports/${id}`),
  exportDownloadUrl: (id: number) => `${API_BASE}/exports/${id}/download`,
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}
