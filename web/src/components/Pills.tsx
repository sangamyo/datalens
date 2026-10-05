import { Icon, type IconName } from './Icon'
import { Spinner } from './Spinner'

type Tone = 'pass' | 'warn' | 'fail' | 'pending' | 'info'

const QC_META: Record<string, { tone: Tone; icon: IconName; label: string }> = {
  pass: { tone: 'pass', icon: 'check', label: 'Pass' },
  warn: { tone: 'warn', icon: 'warn', label: 'Warn' },
  fail: { tone: 'fail', icon: 'x', label: 'Fail' },
  pending: { tone: 'pending', icon: 'clock', label: 'Pending' },
}

/** QC status / severity pill. Color is always paired with an icon + label. */
export function QCPill({ status, title }: { status: string; title?: string }) {
  const meta = QC_META[status] ?? { tone: 'pending' as Tone, icon: 'info' as IconName, label: status }
  return (
    <span className={`pill pill--${meta.tone}`} title={title}>
      <Icon name={meta.icon} size={12} />
      {meta.label}
    </span>
  )
}

const STATUS_META: Record<string, { tone: Tone; label: string; icon?: IconName; busy?: boolean }> = {
  pending: { tone: 'pending', label: 'Queued', busy: true },
  importing: { tone: 'info', label: 'Importing', busy: true },
  building: { tone: 'info', label: 'Building', busy: true },
  running: { tone: 'info', label: 'Running', busy: true },
  ready: { tone: 'pass', label: 'Ready', icon: 'check' },
  failed: { tone: 'fail', label: 'Failed', icon: 'x' },
}

/** Job status pill for datasets & exports. */
export function StatusPill({ status, title }: { status: string; title?: string }) {
  const meta = STATUS_META[status] ?? { tone: 'pending' as Tone, label: status }
  return (
    <span className={`pill pill--${meta.tone}`} title={title}>
      {meta.busy ? <Spinner size={10} /> : meta.icon ? <Icon name={meta.icon} size={12} /> : null}
      {meta.label}
    </span>
  )
}
