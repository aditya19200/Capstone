import { useEffect, useState } from 'react'
import { getDomains } from '../api/client.js'
import { LEGAL_LABELS } from '../api/constants.js'

// The label dropdowns used to read a hardcoded array in constants.js that had
// to be kept in step with the backend by hand. They now read whatever the
// backend says it serves (GET /domains), which is the same domain pack
// model_service reads its id2label from — so a second domain, or a renamed
// label, shows up here without a frontend edit.
//
// LEGAL_LABELS stays as the fallback, and it matters: a reviewer mid-queue
// when the call fails needs a usable dropdown, not an empty one. It is the
// right default because a deployment serving a different domain is exactly
// the case where /domains answers, so falling back to legal is only ever
// wrong when the backend is unreachable anyway.
//
// One fetch per mount, no cache. The label set changes about as often as the
// model is retrained, and the alternative — module-level state shared across
// components — is a worse trade for a list of ten strings.
export default function useLabels() {
  const [labels, setLabels] = useState(LEGAL_LABELS)
  const [activeDomain, setActiveDomain] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await getDomains()
        const active = data.domains?.find((d) => d.key === data.active)
        if (cancelled || !active?.labels?.length) return
        // Sort by id: the dropdown order should match the model's logit
        // order, and the API documents that order but a client should not
        // have to trust it.
        setLabels(
          [...active.labels].sort((a, b) => a.id - b.id).map((l) => l.name),
        )
        setActiveDomain(active)
      } catch (err) {
        // Not fatal — the fallback list is already in state.
        console.error('[useLabels] could not load /domains, using fallback', err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return { labels, activeDomain, loading }
}
