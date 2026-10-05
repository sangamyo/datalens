import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'

export interface Resource<T> {
  data: T | undefined
  error: unknown
  loading: boolean
  /** true until the first request settles */
  initial: boolean
  reload: () => void
  setData: Dispatch<SetStateAction<T | undefined>>
}

/**
 * Fetch data when `deps` change. Keeps the previous data while refetching (no flicker).
 * `poll(data, error)` may return a delay in ms to schedule another fetch, or false to stop.
 * Pass `fetcher = null` to skip fetching.
 */
export function useResource<T>(
  fetcher: (() => Promise<T>) | null,
  deps: unknown[],
  poll?: (data: T | undefined, error: unknown) => number | false,
): Resource<T> {
  const [data, setData] = useState<T | undefined>(undefined)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState<boolean>(fetcher != null)
  const [initial, setInitial] = useState(true)
  const [tick, setTick] = useState(0)

  const fetcherRef = useRef(fetcher)
  const pollRef = useRef(poll)
  useEffect(() => {
    fetcherRef.current = fetcher
    pollRef.current = poll
  })

  useEffect(() => {
    const f = fetcherRef.current
    if (!f) {
      setLoading(false)
      return
    }
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const run = async (background: boolean) => {
      if (!background) setLoading(true)
      let d: T | undefined
      let err: unknown = null
      try {
        d = await (fetcherRef.current ?? f)()
      } catch (e) {
        err = e
      }
      if (cancelled) return
      if (err) setError(err)
      else {
        setError(null)
        setData(d)
      }
      setLoading(false)
      setInitial(false)
      const next = pollRef.current?.(err ? undefined : d, err)
      if (next !== undefined && next !== false) {
        timer = setTimeout(() => run(true), next)
      }
    }
    run(false)
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, initial, reload, setData }
}
