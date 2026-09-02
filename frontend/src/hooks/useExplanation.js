// Module-level cache + poller for SHAP explanations, keyed by prediction_id.
// Lives outside React so a poll in flight survives the ExplainPanel
// unmounting (drawer closed) — reopening the same row reattaches to
// whatever state the background poll has reached instead of re-triggering
// requestExplain or restarting the 3s cadence.

import { useCallback, useSyncExternalStore } from 'react'
import { getExplain, requestExplain } from '../api/client.js'

const POLL_INTERVAL_MS = 3000
const MAX_POLL_MS = 5 * 60 * 1000

// UI-relevant state, keyed by prediction_id. Replaced (not mutated) on each
// update so a stable reference is returned between updates for
// useSyncExternalStore's getSnapshot.
const cache = new Map()

// Polling bookkeeping (interval id, start time) — deliberately separate from
// `cache` so it never triggers a UI notification on its own.
const pollMeta = new Map()

const listeners = new Map() // prediction_id -> Set<() => void>

const DEFAULT_ENTRY = { status: 'idle', tokenImportances: null, generatedAt: null, error: null }

function getEntry(predictionId) {
  if (!cache.has(predictionId)) {
    cache.set(predictionId, DEFAULT_ENTRY)
  }
  return cache.get(predictionId)
}

function notify(predictionId) {
  listeners.get(predictionId)?.forEach((callback) => callback())
}

function setEntry(predictionId, patch) {
  cache.set(predictionId, { ...getEntry(predictionId), ...patch })
  notify(predictionId)
}

function stopPolling(predictionId) {
  const meta = pollMeta.get(predictionId)
  if (meta?.intervalId) {
    clearInterval(meta.intervalId)
  }
  pollMeta.delete(predictionId)
}

async function poll(predictionId) {
  const meta = pollMeta.get(predictionId)
  if (!meta) return // already stopped

  if (Date.now() - meta.startedAt > MAX_POLL_MS) {
    stopPolling(predictionId)
    setEntry(predictionId, { status: 'timeout' })
    return
  }

  try {
    const result = await getExplain(predictionId)

    if (result.status === 'done' || result.status === 'completed') {
      stopPolling(predictionId)
      setEntry(predictionId, {
        status: 'done',
        tokenImportances: result.token_importances,
        generatedAt: result.generated_at,
        error: null,
      })
    } else if (result.status === 'failed') {
      stopPolling(predictionId)
      setEntry(predictionId, { status: 'failed', error: 'Explanation generation failed.' })
    } else {
      // pending / processing
      setEntry(predictionId, { status: result.status })
    }
  } catch (error) {
    console.error('[useExplanation] poll failed', error)
    stopPolling(predictionId)
    setEntry(predictionId, { status: 'failed', error: 'Lost connection while checking explanation status.' })
  }
}

function startPolling(predictionId) {
  if (pollMeta.has(predictionId)) return // already polling
  pollMeta.set(predictionId, {
    startedAt: Date.now(),
    intervalId: setInterval(() => poll(predictionId), POLL_INTERVAL_MS),
  })
}

export async function beginExplanation(predictionId) {
  const entry = getEntry(predictionId)

  // Already have a finished result, or already in flight — don't re-trigger
  // requestExplain. Covering 'requesting' here (not just 'pending'/
  // 'processing') matters because StrictMode double-invokes effects in dev:
  // without it, the second synchronous call would race the first and fire
  // requestExplain twice before it resolves.
  if (entry.status === 'done') return
  if (entry.status === 'requesting' || entry.status === 'pending' || entry.status === 'processing') {
    startPolling(predictionId) // no-op if already polling
    return
  }

  setEntry(predictionId, { status: 'requesting', error: null })

  try {
    await requestExplain(predictionId)
    setEntry(predictionId, { status: 'pending' })
    startPolling(predictionId)
  } catch (error) {
    console.error('[useExplanation] requestExplain failed', error)
    setEntry(predictionId, { status: 'failed', error: 'Could not start the explanation job.' })
  }
}

export function useExplanation(predictionId) {
  const subscribe = useCallback(
    (callback) => {
      if (!predictionId) return () => {}
      if (!listeners.has(predictionId)) listeners.set(predictionId, new Set())
      listeners.get(predictionId).add(callback)
      return () => listeners.get(predictionId)?.delete(callback)
    },
    [predictionId],
  )

  const getSnapshot = useCallback(
    () => (predictionId ? getEntry(predictionId) : DEFAULT_ENTRY),
    [predictionId],
  )

  const entry = useSyncExternalStore(subscribe, getSnapshot)

  const retry = useCallback(() => {
    if (!predictionId) return
    setEntry(predictionId, { status: 'idle', error: null })
    beginExplanation(predictionId)
  }, [predictionId])

  return { ...entry, retry }
}
