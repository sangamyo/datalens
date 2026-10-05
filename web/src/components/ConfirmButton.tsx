import { useEffect, useRef, useState } from 'react'
import { Icon } from './Icon'
import { Spinner } from './Spinner'

/** Two-step in-page confirm (no window.confirm). */
export function ConfirmButton({
  label,
  confirmLabel = 'Delete',
  prompt = 'Are you sure?',
  onConfirm,
  ariaLabel,
  small,
}: {
  label: string
  confirmLabel?: string
  prompt?: string
  onConfirm: () => Promise<void> | void
  ariaLabel?: string
  small?: boolean
}) {
  const [armed, setArmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const cancelRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (armed) cancelRef.current?.focus()
  }, [armed])

  const sm = small ? ' btn--sm' : ''
  if (!armed) {
    return (
      <button
        type="button"
        className={`btn btn--ghost btn--danger${sm}`}
        aria-label={ariaLabel}
        onClick={(e) => {
          e.stopPropagation()
          setArmed(true)
        }}
      >
        <Icon name="trash" size={13} />
        <span className="hide-xs">{label}</span>
      </button>
    )
  }
  return (
    <span
      className="confirm"
      role="group"
      aria-label={prompt}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'Escape') setArmed(false)
      }}
    >
      <span className="muted">{prompt}</span>
      <button
        type="button"
        className={`btn btn--danger-solid${sm}`}
        disabled={busy}
        onClick={async () => {
          setBusy(true)
          try {
            await onConfirm()
          } finally {
            setBusy(false)
            setArmed(false)
          }
        }}
      >
        {busy && <Spinner size={11} />}
        {confirmLabel}
      </button>
      <button ref={cancelRef} type="button" className={`btn${sm}`} onClick={() => setArmed(false)}>
        Cancel
      </button>
    </span>
  )
}
