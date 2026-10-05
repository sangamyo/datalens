import type { ReactNode } from 'react'
import { Icon } from './Icon'

export function ErrorBanner({
  error,
  title = 'Something went wrong',
  onRetry,
  tone = 'error',
  children,
}: {
  error?: unknown
  title?: string
  onRetry?: () => void
  tone?: 'error' | 'warn' | 'info'
  children?: ReactNode
}) {
  const message =
    children ?? (error instanceof Error ? error.message : error != null ? String(error) : null)
  return (
    <div className={`banner banner--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon name={tone === 'info' ? 'info' : 'alert'} size={15} />
      <div className="banner__body">
        <div className="banner__title">{title}</div>
        {message && <div>{message}</div>}
      </div>
      {onRetry && (
        <button type="button" className="btn btn--sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}
