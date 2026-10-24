import useDomain from '../../hooks/useDomain.js'

// Why a confident prediction is in the review queue: the ontology check's
// verdict and the working behind it. Shown on the queue card and again in
// the review drawer, next to the full text it was computed from.
function OntologyFlag({ item }) {
  const { nounPrefix } = useDomain()

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-amber-600 dark:text-amber-300">
          Ontology conflict
        </span>
        <span className="text-sm text-slate-700 dark:text-slate-200">
          The model was confident, but the {nounPrefix}concepts in this text point
          to{' '}
          <strong className="font-semibold">{item.suggested_label}</strong>.
        </span>
      </div>

      {/* The comparison itself, so the flag is checkable rather than a
          verdict the reviewer has to trust. Evidence is the summed weight
          of the concept markers matched for each domain. */}
      {item.evidence_score != null ? (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          <span className="text-slate-500 dark:text-slate-400">Concept evidence:</span>
          <span className="font-mono text-amber-700 dark:text-amber-300">
            {item.suggested_label} {item.evidence_score.toFixed(1)}
          </span>
          <span className="font-mono text-slate-500 dark:text-slate-400">
            {item.predicted_label} {(item.predicted_label_score ?? 0).toFixed(1)}
          </span>
        </div>
      ) : null}

      {item.matched_terms?.length ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-500 dark:text-slate-400">
            Concepts found:
          </span>
          {item.matched_terms.map((term) => (
            <code
              key={term}
              className="rounded bg-slate-500/15 px-1.5 py-0.5 text-xs text-slate-700 dark:text-slate-200"
            >
              {term}
            </code>
          ))}
        </div>
      ) : null}
    </div>
  )
}

export default OntologyFlag
