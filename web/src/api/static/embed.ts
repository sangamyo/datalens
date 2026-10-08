// In-browser query embedding for the static demo: the same model as the API (BAAI/bge-small-en-v1.5, quantized
// ONNX) through transformers.js, both loaded on first use from public CDNs (jsDelivr, Hugging Face Hub) and
// cached by the browser. Matches app/embeddings.py: bge's query instruction, CLS pooling, L2-normalised.

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js'
const MODEL_ID = 'Xenova/bge-small-en-v1.5'
const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

interface Tensor {
  data: Float32Array
}
type Extractor = (text: string, opts: { pooling: 'cls'; normalize: boolean }) => Promise<Tensor>
interface Transformers {
  env: { allowLocalModels: boolean }
  pipeline: (task: 'feature-extraction', model: string, opts: { dtype: string }) => Promise<Extractor>
}

let extractor: Promise<Extractor> | null = null

function loadExtractor(): Promise<Extractor> {
  extractor ??= (async () => {
    const tf = (await import(/* @vite-ignore */ TRANSFORMERS_URL)) as Transformers
    tf.env.allowLocalModels = false
    return tf.pipeline('feature-extraction', MODEL_ID, { dtype: 'q8' })
  })().catch((e) => {
    extractor = null // allow a retry (e.g. after a network error)
    throw e
  })
  return extractor
}

/** Start downloading the model in the background (no-op if already loading). */
export function warmUpQueryModel() {
  loadExtractor().catch(() => {})
}

export async function embedQuery(query: string): Promise<Float32Array> {
  const run = await loadExtractor()
  const out = await run(QUERY_PREFIX + query.trim(), { pooling: 'cls', normalize: true })
  return out.data
}
