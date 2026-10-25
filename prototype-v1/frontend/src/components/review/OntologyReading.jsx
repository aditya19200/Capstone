import useDomain from '../../hooks/useDomain.js'

// What the concept layer sees in the passage, whether or not it raised a
// flag: each label's evidence, strongest first, with the markers it rests on.
// The flag answers one narrow question; a reviewer choosing a label can use
// the rest. It is shown as evidence — it pre-selects nothing — because on
// medical text the markers are noisy (exam shorthand like "shx" counts toward
// Radiology), and a suggestion would steer reviewers wrong.
const SHOWN = 3

function OntologyReading({ reading, predictedLabel }) {
  const { nounPrefix } = useDomain()
  if (!reading) return null

  const rows = reading.labels.slice(0, SHOWN)
  const hidden = reading.labels.length - rows.length
  // One scale for every bar, wide enough to show the flagging bar's tick.
  const scale = Math.max(reading.labels[0]?.score ?? 0, reading.bar_evidence) * 1.1 || 1

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded bg-slate-500/15 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-slate-700">
          Ontology
        </span>
        <span className="text-xs uppercase tracking-[0.16em] text-slate-500">
          what the {nounPrefix}concept layer sees
        </span>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-700">{reading.summary}</p>

      {rows.length ? (
        <ul className="mt-4 space-y-4">
          {rows.map((row) => {
            const own = row.label === predictedLabel
            return (
              <li key={row.label}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm font-medium text-slate-800">
                    {row.label}
                    {own ? (
                      <span className="ml-2 text-xs font-normal text-slate-500">
                        the model&apos;s label
                      </span>
                    ) : null}
                  </span>
                  <span className="shrink-0 font-mono text-xs text-slate-500">
                    {row.score.toFixed(1)} · {row.terms.length}{' '}
                    {row.terms.length === 1 ? 'marker' : 'markers'} ·{' '}
                    {row.strong ? 'strong' : 'light'}
                  </span>
                </div>
                <div className="relative mt-1.5 h-1.5 rounded-full bg-slate-500/15">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(100, (row.score / scale) * 100)}%`,
                      backgroundColor: own ? 'rgb(16, 185, 129)' : 'rgb(245, 158, 11)',
                    }}
                  />
                  <div
                    className="absolute -top-1 h-3.5 w-0.5 rounded bg-slate-500"
                    style={{ left: `${(reading.bar_evidence / scale) * 100}%` }}
                    title={`The check flags at ${reading.bar_evidence}`}
                  />
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {row.terms.slice(0, 8).map((term) => (
                    <code
                      key={term}
                      className="rounded bg-slate-500/15 px-1.5 py-0.5 text-xs text-slate-700"
                    >
                      {term}
                    </code>
                  ))}
                  {row.terms.length > 8 ? (
                    <span className="self-center text-xs text-slate-500">
                      +{row.terms.length - 8} more
                    </span>
                  ) : null}
                </div>
              </li>
            )
          })}
        </ul>
      ) : null}

      {hidden > 0 ? (
        <p className="mt-3 text-xs text-slate-500">
          {hidden} more {hidden === 1 ? 'label has' : 'labels have'} lighter evidence.
        </p>
      ) : null}

      <p className="mt-3 text-xs leading-5 text-slate-500">
        Evidence, not a verdict: the reviewer decides. Markers are words mined from
        the training data that mostly occur under one label, so a generic one (an
        abbreviation, a section heading) can point the wrong way. The tick on each
        bar is where the check flags: {reading.bar_evidence} evidence from{' '}
        {reading.bar_terms} or more markers.
      </p>
    </section>
  )
}

export default OntologyReading
