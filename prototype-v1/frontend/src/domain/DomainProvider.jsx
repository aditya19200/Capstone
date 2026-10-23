import { useEffect, useState } from 'react'
import { getDomains } from '../api/client.js'
import { DomainContext, describe } from '../hooks/useDomain.js'

// Fetches GET /domains once for the whole app. See hooks/useDomain.js for why.
function DomainProvider({ children }) {
  const [value, setValue] = useState(() => ({ ...describe(null), loading: true }))

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await getDomains()
        const active = data.domains?.find((d) => d.key === data.active)
        if (!cancelled) setValue(describe(data.active, active?.display_name))
      } catch (err) {
        // Neutral wording is still correct wording; say nothing domain-specific
        // rather than guess.
        console.error('[DomainProvider] could not load /domains', err)
        if (!cancelled) setValue(describe(null))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return <DomainContext.Provider value={value}>{children}</DomainContext.Provider>
}

export default DomainProvider
