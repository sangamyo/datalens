// Static demo build (VITE_STATIC_DEMO=1): the UI runs on a read-only snapshot in the browser, no backend.
// See deploy/hf-static/README.md.
export const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === '1'

export const REPO_URL = 'https://github.com/sangamyo/datalens'

export const READ_ONLY_MESSAGE = 'Read-only demo — run locally for imports/exports'
