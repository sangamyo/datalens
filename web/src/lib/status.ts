export const isBusyStatus = (s: string | undefined) =>
  s === 'pending' || s === 'importing' || s === 'building' || s === 'running'
