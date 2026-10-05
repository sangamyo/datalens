import type { ReactNode } from 'react'
import { Icon, type IconName } from './Icon'

export function EmptyState({
  icon = 'database',
  title,
  children,
  actions,
}: {
  icon?: IconName
  title: string
  children?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="empty">
      <div className="empty__icon">
        <Icon name={icon} size={20} />
      </div>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {actions && <div className="empty__actions">{actions}</div>}
    </div>
  )
}
