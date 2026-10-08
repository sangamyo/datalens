import { Link } from 'react-router-dom'
import { api, ApiError } from '../../api/client'
import { useResource } from '../../hooks/useResource'
import { ErrorBanner } from '../ErrorBanner'
import { QCPill } from '../Pills'
import { LoadingBlock } from '../Spinner'

const LIMIT = 8

/** Nearest neighbours of a sample by embedding (cosine similarity), within its dataset. */
export function SimilarSamples({ sampleId }: { sampleId: number }) {
  const similar = useResource(() => api.similarSamples(sampleId, LIMIT), [sampleId])
  const items = similar.data

  return (
    <section className="card" aria-labelledby="similar-heading">
      <div className="card__head">
        <div>
          <h2 id="similar-heading">Similar samples</h2>
          <p>Closest by meaning (sentence embeddings, cosine similarity).</p>
        </div>
      </div>
      {similar.initial && similar.loading ? (
        <LoadingBlock label="Finding similar samples…" />
      ) : similar.error instanceof ApiError && similar.error.status === 409 ? (
        <p className="card__body muted" style={{ margin: 0, fontSize: 13 }}>
          This sample has no embedding yet. Embeddings are built in the background after import.
        </p>
      ) : similar.error != null ? (
        <div className="card__body">
          <ErrorBanner title="Could not load similar samples" error={similar.error} onRetry={similar.reload} />
        </div>
      ) : !items?.length ? (
        <p className="card__body muted" style={{ margin: 0, fontSize: 13 }}>
          No other embedded samples in this dataset.
        </p>
      ) : (
        <ul className="similar-list">
          {items.map((it) => (
            <li key={it.id}>
              <Link to={`/samples/${it.id}`} className="similar-item" title={it.prompt_preview}>
                <span className="similar-item__text">{it.prompt_preview || <span className="muted">(empty prompt)</span>}</span>
                <span className="similar-item__meta">
                  <span className="num mono">{it.similarity.toFixed(3)}</span>
                  <QCPill status={it.qc_status} />
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
