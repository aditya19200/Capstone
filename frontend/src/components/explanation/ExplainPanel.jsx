import { useEffect } from 'react'
import { useExplanation, beginExplanation } from '../../hooks/useExplanation.js'
import ExplanationPanel from './ExplanationPanel.jsx'

function Spinner() {
  return (
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600" />
  )
}

function ExplainPanel({ item, onClose }) {
  const predictionId = item?.prediction_id
  const { status, tokenImportances, error, retry } = useExplanation(predictionId)

  useEffect(() => {
    if (predictionId) {
      beginExplanation(predictionId)
    }
  }, [predictionId])

  if (!item) {
    return null
  }

  const shapValues = (tokenImportances || []).map(({ token, importance }) => ({
    token,
    value: importance,
  }))

  return (
    <>
      <button
        type="button"
        aria-label="Close explanation panel backdrop"
        className="fixed inset-0 z-30 bg-slate-950/30 backdrop-blur-sm"
        onClick={onClose}
      />

      <aside className="fixed bottom-0 right-0 top-20 z-40 w-full max-w-2xl overflow-y-auto border-l border-slate-200 bg-slate-50 shadow-2xl">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/95 px-6 py-5 backdrop-blur">
          <div>
            <p className="section-kicker">Explanation</p>
            <h3 className="mt-2 text-xl font-semibold text-slate-900">{item.predicted_label}</h3>
          </div>
          <button type="button" onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>

        <div className="space-y-6 p-6">
          {(status === 'requesting' || status === 'pending' || status === 'processing' || status === 'idle') && (
            <div className="dashboard-card flex flex-col items-center gap-4 py-12 text-center">
              <Spinner />
              <p className="text-sm font-medium text-slate-700">
                Generating explanation (usually 30-90s)...
              </p>
              <p className="text-xs uppercase tracking-[0.16em] text-slate-400">
                This runs SHAP on the model — it's slow by design, not stuck.
              </p>
            </div>
          )}

          {status === 'timeout' && (
            <div className="dashboard-card border-dashed text-center">
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
                Taking longer than expected
              </p>
              <p className="mt-3 text-sm text-slate-600">
                This explanation has been generating for a while. It may still complete — you can
                check again.
              </p>
              <button type="button" onClick={retry} className="btn-primary mt-4 px-6 py-3">
                Check Again
              </button>
            </div>
          )}

          {status === 'failed' && (
            <div className="dashboard-card border-dashed text-center">
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-red-600">
                Explanation failed
              </p>
              <p className="mt-3 text-sm text-slate-600">
                {error || 'Something went wrong generating this explanation.'}
              </p>
              <button type="button" onClick={retry} className="btn-primary mt-4 px-6 py-3">
                Retry
              </button>
            </div>
          )}

          {status === 'done' && <ExplanationPanel shapValues={shapValues} />}
        </div>
      </aside>
    </>
  )
}

export default ExplainPanel
