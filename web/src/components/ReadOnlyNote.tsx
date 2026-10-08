import { READ_ONLY_MESSAGE, REPO_URL } from '../lib/demo'
import { Icon } from './Icon'

/** Shown next to actions that are disabled in the static demo build (import, QC, embeddings, export). */
export function ReadOnlyNote() {
  return (
    <span className="readonly-note">
      <Icon name="info" size={12} />
      <span>
        {READ_ONLY_MESSAGE} ·{' '}
        <a href={REPO_URL} target="_blank" rel="noreferrer">
          GitHub
        </a>
      </span>
    </span>
  )
}
