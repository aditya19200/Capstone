import { useMemo, useState } from 'react'

function OntologyNodeCard({ nodeName, connections }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition hover:border-indigo-200 hover:shadow-md">
      <button
        type="button"
        onClick={() => setIsExpanded((currentValue) => !currentValue)}
        className="flex w-full items-center justify-between gap-4 text-left"
      >
        <div>
          <p className="text-sm font-semibold text-slate-900">{nodeName}</p>
          <p className="mt-1 text-xs uppercase tracking-[0.16em] text-slate-500">
            {connections.length} linked concepts
          </p>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
          {isExpanded ? 'Hide' : 'Expand'}
        </span>
      </button>

      <div
        className={[
          'grid overflow-hidden transition-all duration-300',
          isExpanded ? 'mt-4 grid-rows-[1fr]' : 'grid-rows-[0fr]',
        ].join(' ')}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4">
            {connections.map((connection) => (
              <span
                key={`${nodeName}-${connection}`}
                className="rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
              >
                {connection}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function OntologyGraphViewer({ selectedConcept, ontologyData }) {
  const relatedNodes = ontologyData[selectedConcept] || []

  const graphCards = useMemo(
    () =>
      relatedNodes.map((nodeName) => ({
        nodeName,
        connections: ontologyData[nodeName] || [],
      })),
    [ontologyData, relatedNodes],
  )

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-col gap-3 border-b border-slate-100 pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
            Relationship Panel
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-slate-900">{selectedConcept}</h3>
        </div>
        <div className="rounded-full bg-indigo-50 px-4 py-2 text-sm font-medium text-indigo-700">
          {relatedNodes.length} related nodes
        </div>
      </div>

      <div className="mt-6 space-y-6">
        <div className="rounded-2xl bg-slate-50 p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            Direct Relationships
          </p>
          <div className="mt-4 flex flex-wrap gap-3">
            {relatedNodes.map((nodeName) => (
              <div
                key={`${selectedConcept}-${nodeName}`}
                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm"
              >
                <p className="text-sm font-semibold text-slate-900">{nodeName}</p>
                <p className="mt-1 text-xs uppercase tracking-[0.16em] text-slate-500">
                  Connected to {selectedConcept}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-2xl bg-slate-50 p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                Expandable Nodes
              </p>
              <p className="mt-2 text-sm text-slate-600">
                Open each node to inspect the next layer of legal concept connections.
              </p>
            </div>
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            {graphCards.map((card) => (
              <OntologyNodeCard
                key={card.nodeName}
                nodeName={card.nodeName}
                connections={card.connections}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export default OntologyGraphViewer
