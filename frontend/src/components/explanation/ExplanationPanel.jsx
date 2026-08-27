import ShapBarChart from './ShapBarChart.jsx'
import TokenHighlighter from './TokenHighlighter.jsx'

function ExplanationPanel({ summary, shapValues }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="border-b border-slate-100 pb-5">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          Explanation
        </p>
        <h3 className="mt-2 text-xl font-semibold text-slate-900">Why the model chose this label</h3>
      </div>

      <div className="mt-6 space-y-6">
        <div className="rounded-2xl bg-indigo-50 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-indigo-700">
            Summary
          </p>
          <p className="mt-2 text-sm leading-7 text-slate-700">{summary}</p>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            Token Highlighting
          </p>
          <div className="mt-3">
            <TokenHighlighter tokens={shapValues} />
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            SHAP Importance
          </p>
          <div className="mt-3 rounded-2xl bg-slate-50 p-4">
            <ShapBarChart tokens={shapValues} />
          </div>
        </div>
      </div>
    </section>
  )
}

export default ExplanationPanel
