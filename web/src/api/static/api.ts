// The API client of the static demo build: every endpoint the UI uses, answered in the browser from a read-only
// snapshot (deploy/hf-static/export_snapshot.py writes it to data/). Filters, ordering, NL search and cosine
// ranking port the backend's logic (app/queries.py, routers/search.py, routers/semantic.py), so the UI cannot
// tell the difference. Anything that would write (import, QC, embeddings, exports) fails with a read-only error.
import type { Api } from '../client'
import { ApiError } from '../errors'
import type {
  DatasetDetail,
  DatasetOut,
  EmbeddingStatus,
  ExportOut,
  QCSummary,
  SampleDetail,
  SampleFilter,
  SampleList,
  SampleOut,
  SearchRequest,
  SearchResponse,
  SemanticSearchRequest,
  SemanticSearchResponse,
  SimilarSample,
} from '../types'
import { READ_ONLY_MESSAGE } from '../../lib/demo'
import { embedQuery } from './embed'

// ---------------------------------------------------------------- snapshot files
const DATA_ROOT = 'data/'

const cache = new Map<string, Promise<unknown>>()

function fetchData<T>(path: string, kind: 'json' | 'buffer' = 'json'): Promise<T> {
  let p = cache.get(path) as Promise<T> | undefined
  if (!p) {
    p = (async () => {
      let res: Response
      try {
        res = await fetch(new URL(DATA_ROOT + path, document.baseURI))
      } catch {
        throw new ApiError(0, 'Could not load the demo snapshot. Check your connection and retry.')
      }
      if (!res.ok) throw new ApiError(res.status, `Could not load the demo snapshot (${res.status} for ${path})`)
      return (kind === 'json' ? res.json() : res.arrayBuffer()) as Promise<T>
    })()
    p.catch(() => cache.delete(path)) // retry on the next call
    cache.set(path, p)
  }
  return p
}

interface IndexFile {
  chunk_size: number
  columns: string[]
  rows: unknown[][]
}

interface Row {
  out: SampleOut
  /** checks with a warn/fail result */
  flagged: string[]
}

interface DatasetIndex {
  rows: Row[] // ordered by sample_index
  pos: Map<number, number> // sample id -> position in rows
  chunkSize: number
}

interface Vectors {
  dim: number
  data: Float32Array
  norms: Float64Array
  slot: Map<number, number> // sample id -> vector number
}

const dsPath = (id: number, file: string) => `datasets/${id}/${file}`

const listDatasets = () => fetchData<DatasetOut[]>('datasets.json')

async function ensureDataset(id: number): Promise<DatasetOut> {
  const ds = (await listDatasets()).find((d) => d.id === id)
  if (!ds) throw new ApiError(404, `dataset ${id} not found`)
  return ds
}

const indexes = new Map<number, Promise<DatasetIndex>>()

function loadIndex(datasetId: number): Promise<DatasetIndex> {
  let p = indexes.get(datasetId)
  if (!p) {
    p = fetchData<IndexFile>(dsPath(datasetId, 'index.json')).then((f) => {
      const col = Object.fromEntries(f.columns.map((c, i) => [c, i]))
      const rows = f.rows.map((r): Row => {
        const v = (c: string) => r[col[c]]
        return {
          out: {
            id: v('id') as number,
            dataset_id: datasetId,
            sample_index: v('sample_index') as number,
            category: v('category') as string | null,
            prompt_preview: v('prompt_preview') as string,
            tokens_est: v('tokens_est') as number,
            lang: v('lang') as string | null,
            qc_status: v('qc_status') as string,
            qc_score: v('qc_score') as number | null,
          },
          flagged: v('flagged') as string[],
        }
      })
      return { rows, pos: new Map(rows.map((r, i) => [r.out.id, i])), chunkSize: f.chunk_size }
    })
    p.catch(() => indexes.delete(datasetId))
    indexes.set(datasetId, p)
  }
  return p
}

const loadChunk = (datasetId: number, n: number) => fetchData<SampleDetail[]>(dsPath(datasetId, `samples/${n}.json`))

/** Full sample records of a dataset by position (needed for text search). */
async function loadAllDetails(datasetId: number): Promise<SampleDetail[]> {
  const idx = await loadIndex(datasetId)
  const n = Math.ceil(idx.rows.length / idx.chunkSize)
  return (await Promise.all(Array.from({ length: n }, (_, i) => loadChunk(datasetId, i)))).flat()
}

async function findSample(sampleId: number): Promise<{ datasetId: number; idx: DatasetIndex; pos: number }> {
  for (const ds of await listDatasets()) {
    const idx = await loadIndex(ds.id)
    const pos = idx.pos.get(sampleId)
    if (pos !== undefined) return { datasetId: ds.id, idx, pos }
  }
  throw new ApiError(404, `sample ${sampleId} not found`)
}

const vectorsCache = new Map<number, Promise<Vectors>>()

function loadVectors(datasetId: number): Promise<Vectors> {
  let p = vectorsCache.get(datasetId)
  if (!p) {
    p = Promise.all([
      fetchData<{ dim: number; ids: number[] }>(dsPath(datasetId, 'vectors.json')),
      fetchData<ArrayBuffer>(dsPath(datasetId, 'vectors.bin'), 'buffer'),
    ]).then(([meta, buf]) => {
      const data = new Float32Array(buf)
      const norms = new Float64Array(meta.ids.length)
      for (let i = 0; i < meta.ids.length; i++) norms[i] = norm(data, i * meta.dim, meta.dim)
      return { dim: meta.dim, data, norms, slot: new Map(meta.ids.map((id, i) => [id, i])) }
    })
    p.catch(() => vectorsCache.delete(datasetId))
    vectorsCache.set(datasetId, p)
  }
  return p
}

function norm(v: ArrayLike<number>, offset: number, dim: number): number {
  let s = 0
  for (let i = 0; i < dim; i++) s += v[offset + i] * v[offset + i]
  return Math.sqrt(s)
}

// ---------------------------------------------------------------- filters (app/queries.py apply_sample_filter)
/** SQL (I)LIKE pattern -> RegExp: % = any run, _ = any char, backslash escapes. */
function likeRegex(pattern: string): RegExp {
  let re = ''
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i]
    if (c === '\\' && i + 1 < pattern.length) re += escapeRe(pattern[++i])
    else if (c === '%') re += '[\\s\\S]*'
    else if (c === '_') re += '[\\s\\S]'
    else re += escapeRe(c)
  }
  return new RegExp(`^${re}$`, 'iu')
}

const escapeRe = (c: string) => c.replace(/[\\^$.*+?()[\]{}|/]/g, '\\$&')

/** Rows of one dataset matching `f` (its dataset_id is ignored), in sample_index order. */
async function filterRows(datasetId: number, f: SampleFilter): Promise<Row[]> {
  const idx = await loadIndex(datasetId)
  const category = f.category ? likeRegex(f.category) : null
  const text = f.text_contains ? likeRegex(`%${f.text_contains}%`) : null
  const details = text ? await loadAllDetails(datasetId) : null
  const statuses = f.qc_status?.length ? new Set<string>(f.qc_status) : null
  return idx.rows.filter(({ out: s, flagged }, i) => {
    if (statuses && !statuses.has(s.qc_status)) return false
    if (category && (s.category == null || !category.test(s.category))) return false
    if (f.lang && s.lang !== f.lang) return false
    if (f.min_tokens != null && s.tokens_est < f.min_tokens) return false
    if (f.max_tokens != null && s.tokens_est > f.max_tokens) return false
    if (text) {
      const d = details![i]
      if (!text.test(d.prompt) && !(d.context != null && text.test(d.context)) && !text.test(d.response)) return false
    }
    if (f.failed_check && !flagged.includes(f.failed_check)) return false
    return true
  })
}

/** app/queries.py find_samples: matching rows ordered by (dataset_id, sample_index), plus the total. */
async function findSamples(f: SampleFilter, limit: number, offset: number): Promise<SampleList> {
  const ids = f.dataset_id != null ? [f.dataset_id] : (await listDatasets()).map((d) => d.id).sort((a, b) => a - b)
  const rows = (await Promise.all(ids.map((id) => filterRows(id, f)))).flat()
  return { items: rows.slice(offset, offset + limit).map((r) => ({ ...r.out })), total: rows.length }
}

/** app/queries.py nearest_samples: exact cosine ranking (the API's HNSW index has recall 1.0 here). */
async function nearestSamples(
  datasetId: number,
  query: ArrayLike<number>,
  f: SampleFilter,
  limit: number,
  excludeId?: number,
): Promise<SimilarSample[]> {
  const [vec, rows] = await Promise.all([loadVectors(datasetId), filterRows(datasetId, f)])
  const qn = norm(query, 0, vec.dim)
  const scored: { row: Row; sim: number }[] = []
  for (const row of rows) {
    const slot = vec.slot.get(row.out.id)
    if (slot === undefined || row.out.id === excludeId) continue
    const off = slot * vec.dim
    let dot = 0
    for (let i = 0; i < vec.dim; i++) dot += vec.data[off + i] * query[i]
    scored.push({ row, sim: dot / (vec.norms[slot] * qn) })
  }
  scored.sort((a, b) => b.sim - a.sim)
  return scored.slice(0, limit).map(({ row, sim }) => ({ ...row.out, similarity: Math.round(sim * 1e4) / 1e4 }))
}

function checkRange(name: string, v: number, lo: number, hi: number) {
  if (!Number.isInteger(v) || v < lo || v > hi) throw new ApiError(422, `${name}: must be between ${lo} and ${hi}`)
}

function checkQuery(q: string) {
  if (q.length < 1 || q.length > 500) throw new ApiError(422, 'query: must be 1 to 500 characters')
}

const readOnly = (): never => {
  throw new ApiError(405, READ_ONLY_MESSAGE)
}

// ---------------------------------------------------------------- endpoints
export const staticApi: Api = {
  // datasets
  listDatasets: async () => structuredClone(await listDatasets()),
  getDataset: async (id) => {
    await ensureDataset(id)
    return structuredClone(await fetchData<DatasetDetail>(dsPath(id, 'detail.json')))
  },
  importDataset: async () => readOnly(),
  deleteDataset: async () => readOnly(),

  // samples
  listSamples: async (datasetId, filter, limit = 50, offset = 0) => {
    await ensureDataset(datasetId)
    checkRange('limit', limit, 1, 500)
    return findSamples({ ...filter, dataset_id: datasetId }, limit, Math.max(0, offset))
  },
  getSample: async (id) => {
    const { datasetId, idx, pos } = await findSample(id)
    const chunk = await loadChunk(datasetId, Math.floor(pos / idx.chunkSize))
    return structuredClone(chunk[pos % idx.chunkSize])
  },

  // QC
  runDatasetQC: async () => readOnly(),
  runSampleQC: async () => readOnly(),
  getSampleQC: async (id) => (await staticApi.getSample(id)).qc_results,
  getQCSummary: async (id) => {
    await ensureDataset(id)
    return structuredClone(await fetchData<QCSummary>(dsPath(id, 'qc-summary.json')))
  },

  // search (routers/search.py): the rule-based parser; no LLM in the static build
  search: async (body: SearchRequest): Promise<SearchResponse> => {
    checkQuery(body.query)
    const datasetId = body.dataset_id ?? null
    if (datasetId != null) await ensureDataset(datasetId)
    // loaded on first use: its module-level regexes would otherwise keep it in normal builds' bundle
    const { parseRules, resolveCategory } = await import('./nlsearch')
    const f = parseRules(body.query, datasetId)
    if (f.category) {
      // canonical names ("brainstorming") -> the dataset's own spelling ("Brainstorm")
      const ids = datasetId != null ? [datasetId] : (await listDatasets()).map((d) => d.id)
      const names = new Set<string>()
      for (const id of ids) for (const r of (await loadIndex(id)).rows) if (r.out.category != null) names.add(r.out.category)
      f.category = resolveCategory(f.category, names)
    }
    const { items, total } = await findSamples(f, body.limit ?? 50, 0)
    return { filter: f, parser: 'rules', items, total }
  },

  // semantic search (routers/semantic.py)
  getEmbeddingStatus: async (datasetId) => {
    await ensureDataset(datasetId)
    return structuredClone(await fetchData<EmbeddingStatus>(dsPath(datasetId, 'embeddings.json')))
  },
  embedDataset: async () => readOnly(),
  semanticSearch: async (body: SemanticSearchRequest): Promise<SemanticSearchResponse> => {
    checkQuery(body.query)
    const limit = body.limit ?? 25
    checkRange('limit', limit, 1, 100)
    await ensureDataset(body.dataset_id)
    const status = await fetchData<EmbeddingStatus>(dsPath(body.dataset_id, 'embeddings.json'))
    if (status.embedded === 0) throw new ApiError(409, 'this dataset has no embeddings yet')
    let vector: Float32Array
    try {
      vector = await embedQuery(body.query)
    } catch (e) {
      throw new ApiError(503, `embedding model unavailable (${e instanceof Error ? e.message : String(e)})`)
    }
    const items = await nearestSamples(body.dataset_id, vector, { ...body.filter, dataset_id: body.dataset_id }, limit)
    return { model: status.model, items }
  },
  similarSamples: async (sampleId, limit = 8) => {
    checkRange('limit', limit, 1, 50)
    const { datasetId } = await findSample(sampleId)
    const vec = await loadVectors(datasetId)
    const slot = vec.slot.get(sampleId)
    if (slot === undefined) throw new ApiError(409, `sample ${sampleId} has no embedding yet`)
    const v = vec.data.subarray(slot * vec.dim, (slot + 1) * vec.dim)
    return nearestSamples(datasetId, v, { dataset_id: datasetId }, limit, sampleId)
  },

  // exports: none in the snapshot, and creating one needs the worker
  createExport: async () => readOnly(),
  listExports: async (): Promise<ExportOut[]> => [],
  getExport: async (id) => {
    throw new ApiError(404, `export ${id} not found`)
  },
  exportDownloadUrl: () => '#',
}
