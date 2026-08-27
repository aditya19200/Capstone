function KnowledgeGraphPanel({ concepts }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="border-b border-slate-100 pb-5">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          Knowledge Graph
        </p>
        <h3 className="mt-2 text-xl font-semibold text-slate-900">Related legal concepts</h3>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {concepts.map((concept) => (
          <div
            key={concept.name}
            className="rounded-2xl border border-indigo-100 bg-indigo-50 px-4 py-3"
          >
            <p className="text-sm font-semibold text-indigo-900">{concept.name}</p>
            <p className="mt-1 text-xs uppercase tracking-[0.16em] text-indigo-700">
              {concept.relation}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}

export default KnowledgeGraphPanel
