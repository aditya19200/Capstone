import useDomain from '../../hooks/useDomain.js'

// Puts the model's own explanation next to the domain ontology's reading of
// the same text. SHAP says which words moved the model; the concept markers
// say which domain those words belong to. Neither answers "did it reason
// about the right things" alone — together they do, and a reviewer can
// check it rather than trust it.
//
// Shown inside the explanation panel, under the token chart it describes.

const VERDICTS = {
  conflict: {
    label: 'Conflict',
    tone: 'border-red-500/40 bg-red-500/10',
    chip: 'bg-red-500/20 text-red-700 dark:text-red-300',
  },
  mixed: {
    label: 'Mixed evidence',
    tone: 'border-amber-500/40 bg-amber-500/10',
    chip: 'bg-amber-500/20 text-amber-700 dark:text-amber-300',
  },
  overruled: {
    label: 'Contrary evidence overruled',
    tone: 'border-amber-500/40 bg-amber-500/10',
    chip: 'bg-amber-500/20 text-amber-700 dark:text-amber-300',
  },
  aligned: {
    label: 'Agrees',
    tone: 'border-emerald-500/40 bg-emerald-500/10',
    chip: 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300',
  },
  no_overlap: {
    label: 'No common ground',
    tone: 'border-slate-400/40 bg-slate-500/10',
    chip: 'bg-slate-500/20 text-slate-700 dark:text-slate-200',
  },
}

function TermRow({ row }) {
  const toward = row.direction === 'toward'
  return (
    <li className="flex items-center justify-between gap-3 py-1">
      <div className="flex min-w-0 items-center gap-2">
        <code className="rounded bg-slate-500/15 px-1.5 py-0.5 text-xs text-slate-800 dark:text-slate-100">
          {row.term}
        </code>
        <span className="truncate text-xs text-slate-500 dark:text-slate-400">{row.domain}</span>
      </div>
      <span
        className={`shrink-0 font-mono text-xs tabular-nums ${
          toward ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-700 dark:text-red-300'
        }`}
        // The sign is the whole point: a term can be present and still have
        // argued against the label the model chose.
        title={toward ? 'pushed toward the predicted label' : 'pushed against the predicted label'}
      >
        {toward ? '→' : '←'} {row.shap_weight > 0 ? '+' : ''}
        {row.shap_weight.toFixed(3)}
      </span>
    </li>
  )
}

function AgreementCard({ agreement }) {
  const { nounPrefix } = useDomain()
  if (!agreement) return null
  const v = VERDICTS[agreement.verdict] ?? VERDICTS.no_overlap
  const rows = [...(agreement.supporting || []), ...(agreement.contradicting || [])]

  return (
    <section className={`rounded-xl border p-4 ${v.tone}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${v.chip}`}
        >
          {v.label}
        </span>
        <span className="text-xs uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">
          model&apos;s reasoning vs {nounPrefix}ontology
        </span>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-700 dark:text-slate-200">
        {agreement.summary}
      </p>

      {rows.length > 0 ? (
        <ul className="mt-3 divide-y divide-slate-500/10 border-t border-slate-500/10 pt-1">
          {rows.map((row) => (
            <TermRow key={`${row.domain}-${row.term}`} row={row} />
          ))}
        </ul>
      ) : null}

      {/* The ontology's own tally, independent of what the model looked at —
          so a reviewer can see the concept evidence even when none of it
          overlaps the model's attention. */}
      {agreement.rival_label ? (
        <p className="mt-3 flex flex-wrap gap-x-4 text-xs text-slate-500 dark:text-slate-400">
          <span>
            Concept evidence — {agreement.predicted_label}:{' '}
            <span className="font-mono">{agreement.predicted_evidence?.toFixed(1)}</span>
          </span>
          <span>
            {agreement.rival_label}:{' '}
            <span className="font-mono">{agreement.rival_evidence?.toFixed(1)}</span>
          </span>
        </p>
      ) : null}

      <p className="mt-3 text-xs leading-5 text-slate-500 dark:text-slate-400">
        This compares two signals; it does not decide anything. Matching is by
        vocabulary, so a passage that mentions another {nounPrefix}field&apos;s terms
        without being about it can trip it.
      </p>
    </section>
  )
}

export default AgreementCard
