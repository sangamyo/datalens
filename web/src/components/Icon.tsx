import type { SVGProps } from 'react'

const PATHS = {
  check: 'M3.5 8.5l3 3 6-7',
  warn: 'M8 2.2L14.5 13.5h-13L8 2.2zM8 6.5v3M8 11.6v.1',
  x: 'M4 4l8 8M12 4l-8 8',
  clock: 'M8 2a6 6 0 100 12A6 6 0 008 2zM8 5v3.2l2 1.3',
  alert: 'M8 2a6 6 0 100 12A6 6 0 008 2zM8 5v3.5M8 10.8v.1',
  info: 'M8 2a6 6 0 100 12A6 6 0 008 2zM8 7.2v3.8M8 5.1v.1',
  search: 'M7 2.5a4.5 4.5 0 110 9 4.5 4.5 0 010-9zM10.4 10.4L14 14',
  database: 'M2.5 4c0-1.1 2.5-2 5.5-2s5.5.9 5.5 2-2.5 2-5.5 2-5.5-.9-5.5-2zM2.5 4v8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V4M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2',
  download: 'M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10',
  refresh: 'M13 8a5 5 0 11-1.5-3.6M13 2.5v3h-3',
  trash: 'M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5',
  chevronLeft: 'M10 3.5L5.5 8l4.5 4.5',
  chevronRight: 'M6 3.5L10.5 8 6 12.5',
  external: 'M9.5 2.5h4v4M13.5 2.5L8 8M12 9.5v4H2.5V4h4',
  tag: 'M2.5 2.5h5l6 6-5 5-6-6zM5.3 5.3v.1',
  link: 'M6.8 9.2l2.4-2.4M7.4 4.6l1.2-1.2a2.6 2.6 0 013.7 3.7l-1.2 1.2M8.6 11.4l-1.2 1.2a2.6 2.6 0 01-3.7-3.7l1.2-1.2',
  shield: 'M8 1.8l5 2v4c0 3-2.2 5.3-5 6.4-2.8-1.1-5-3.4-5-6.4v-4z',
  package: 'M8 1.8l5.5 3v6.4L8 14.2l-5.5-3V4.8zM2.5 4.8L8 7.8l5.5-3M8 7.8v6.4',
  sparkle: 'M8 2l1.4 4.6L14 8l-4.6 1.4L8 14l-1.4-4.6L2 8l4.6-1.4z',
  filter: 'M2.5 3h11l-4.2 5.2V13l-2.6-1.2V8.2z',
  chart: 'M2.5 2.5v11h11M5 10l2.5-3 2 2 3.5-4.5',
} as const

export type IconName = keyof typeof PATHS

export function Icon({
  name,
  size = 14,
  ...rest
}: { name: IconName; size?: number } & Omit<SVGProps<SVGSVGElement>, 'name'>) {
  const filled = name === 'sparkle'
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth={filled ? 0 : 1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  )
}

/** DataLens mark: a magnifying lens over data rows. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--accent)" />
      <path d="M7 10h8M7 15h5M7 20h4" stroke="var(--accent-text)" strokeOpacity="0.55" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="18.5" cy="15.5" r="5.5" fill="none" stroke="var(--accent-text)" strokeWidth="2.6" />
      <path d="M22.6 19.6l3.6 3.6" stroke="var(--accent-text)" strokeWidth="2.8" strokeLinecap="round" />
    </svg>
  )
}
