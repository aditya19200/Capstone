import { useMemo, useState } from 'react'
import OntologyGraphViewer from '../components/ontology/OntologyGraphViewer.jsx'

const ontologyData = {
  'Contract Law': ['Agreement', 'Obligation', 'Breach', 'Consideration'],
  'Criminal Law': ['Offense', 'Punishment', 'Evidence', 'Mens Rea'],
  'Property Law': ['Title', 'Possession', 'Lease', 'Transfer'],
  Agreement: ['Offer', 'Acceptance', 'Mutual Assent'],
  Obligation: ['Performance', 'Duty', 'Remedy'],
  Breach: ['Damages', 'Specific Performance', 'Termination'],
  Consideration: ['Promise', 'Exchange', 'Value'],
  Offense: ['Actus Reus', 'Mens Rea', 'Liability'],
  Punishment: ['Sentencing', 'Fine', 'Imprisonment'],
  Evidence: ['Witness', 'Documentary Proof', 'Burden of Proof'],
  'Mens Rea': ['Intent', 'Knowledge', 'Recklessness'],
  Title: ['Ownership', 'Registration', 'Encumbrance'],
  Possession: ['Occupancy', 'Control', 'Adverse Possession'],
  Lease: ['Tenancy', 'Rent', 'Covenant'],
  Transfer: ['Assignment', 'Conveyance', 'Registration'],
}

const rootConcepts = ['Contract Law', 'Criminal Law', 'Property Law']

function OntologyPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedConcept, setSelectedConcept] = useState(rootConcepts[0])

  const filteredConcepts = useMemo(
    () =>
      rootConcepts.filter((concept) =>
        concept.toLowerCase().includes(searchTerm.trim().toLowerCase()),
      ),
    [searchTerm],
  )

  const activeConcept =
    filteredConcepts.find((concept) => concept === selectedConcept) || filteredConcepts[0] || null

  return (
    <section className="space-y-6">
      <div className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Ontology Explorer
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Legal Knowledge Graph
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Explore relationships between legal concepts and entities.
        </p>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <label
          htmlFor="ontology-search"
          className="mb-3 block text-sm font-semibold uppercase tracking-[0.2em] text-slate-500"
        >
          Search Concepts
        </label>
        <input
          id="ontology-search"
          type="text"
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.target.value)}
          placeholder="Search legal concepts (e.g. Contract Law)"
          className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:bg-white focus:ring-4 focus:ring-indigo-100"
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[320px,minmax(0,1fr)]">
        <aside className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="border-b border-slate-100 pb-5">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
              Concept Panel
            </p>
            <h3 className="mt-2 text-xl font-semibold text-slate-900">Legal concepts</h3>
          </div>

          <div className="mt-5 max-h-[620px] space-y-3 overflow-y-auto pr-1">
            {filteredConcepts.length ? (
              filteredConcepts.map((concept) => (
                <button
                  key={concept}
                  type="button"
                  onClick={() => setSelectedConcept(concept)}
                  className={[
                    'w-full rounded-2xl border px-4 py-4 text-left transition',
                    activeConcept === concept
                      ? 'border-indigo-200 bg-indigo-50 shadow-sm'
                      : 'border-slate-200 bg-white hover:border-indigo-200 hover:bg-slate-50',
                  ].join(' ')}
                >
                  <p className="text-sm font-semibold text-slate-900">{concept}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.16em] text-slate-500">
                    {ontologyData[concept]?.length || 0} related nodes
                  </p>
                </button>
              ))
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-600">
                No concepts match your search.
              </div>
            )}
          </div>
        </aside>

        <div className="min-w-0">
          {activeConcept ? (
            <OntologyGraphViewer selectedConcept={activeConcept} ontologyData={ontologyData} />
          ) : (
            <div className="rounded-3xl border border-dashed border-slate-300 bg-white p-10 text-center shadow-sm">
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
                Relationship Panel
              </p>
              <p className="mt-3 text-sm text-slate-600">
                Search for a legal concept to begin exploring its related nodes.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

export default OntologyPage
