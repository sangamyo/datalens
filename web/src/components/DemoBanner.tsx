import { REPO_URL } from '../lib/demo'
import { Icon } from './Icon'

/** Thin strip above the header of the static demo build. */
export function DemoBanner() {
  return (
    <div className="demo-banner" role="note">
      <div className="demo-banner__inner">
        <span className="pill pill--info">Static demo</span>
        <span>
          Read-only snapshot with precomputed QC results and embeddings — search runs in your browser.
        </span>
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="demo-banner__link">
          Source on GitHub
          <Icon name="external" size={12} />
        </a>
      </div>
    </div>
  )
}
