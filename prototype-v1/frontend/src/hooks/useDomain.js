import { createContext, useContext } from 'react'

// The domain this deployment serves, for wording.
//
// The UI used to say "legal" in twelve places — the header on every page, the
// sidebar, the Annotate page, the Metrics chart — so the medical instance on
// :5181 introduced itself as a "Legal Annotation System" while classifying
// cardiology reports. The words now come from the same domain pack the model's
// labels do (GET /domains), so a second domain needs no frontend edit.
//
// Kept apart from DomainProvider.jsx so neither file mixes a component with
// other exports, which is what react-refresh/only-export-components objects to.
export const DomainContext = createContext(null)

const UNKNOWN = {
  key: null,
  label: null,
  noun: null,
  // Prefixes rather than bare words, so a sentence still reads correctly in
  // the moment before /domains answers, or if it never does:
  // `${labelPrefix}Annotation System` is "Annotation System", not
  // "null Annotation System".
  labelPrefix: '',
  nounPrefix: '',
  displayName: null,
  loading: true,
}

export function describe(key, displayName) {
  if (!key) return { ...UNKNOWN, loading: false }
  const label = key.charAt(0).toUpperCase() + key.slice(1)
  return {
    key,
    label, // "Legal" / "Medical" — matches the version table's naming
    noun: key, // "legal" / "medical" — for running prose
    labelPrefix: `${label} `,
    nounPrefix: `${key} `,
    displayName: displayName ?? label,
    loading: false,
  }
}

export default function useDomain() {
  return useContext(DomainContext) ?? UNKNOWN
}
