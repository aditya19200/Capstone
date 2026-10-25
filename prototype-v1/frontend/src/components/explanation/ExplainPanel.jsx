import { useEffect, useState } from 'react'
import { getOntologyReading } from '../../api/client.js'
import { useExplanation, beginExplanation } from '../../hooks/useExplanation.js'
import ConfidenceBadge from '../prediction/ConfidenceBadge.jsx'
import ConceptText from '../review/ConceptText.jsx'
import OntologyFlag from '../review/OntologyFlag.jsx'
import OntologyReading from '../review/OntologyReading.jsx'
import AgreementCard from './AgreementCard.jsx'
import ExplanationPanel from './ExplanationPanel.jsx'

function Spinner() {
  return (
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600" />
  )
}

function ExplainPanel({ item, onClose }) {
  const predictionId = item?.prediction_id
  const { status, tokenImportances, agreement, error, retry } = useExplanation(predictionId)

  useEffect(() => {
    if (predictionId) {
      beginExplanation(predictionId)
    }
  }, [predictionId])

  // The concept layer's reading is plain matching, back in milliseconds, so
  // it is there to read while SHAP runs. Stored with the id it belongs to, so
  // a drawer reopened on another item never shows the previous one's.
  const text = item?.text_content
  const predictedLabel = item?.predicted_label
  const [reading, setReading] = useState({ id: null, data: null })
  useEffect(() => {
    if (!predictionId || !text) return undefined
    let cancelled = false
    getOntologyReading({ text, predictedLabel })
      .then((data) => {
        if (!cancelled) setReading({ id: predictionId, data })
      })
      .catch(() => {}) // optional: the drawer reads fine without it
    return () => {
      cancelled = true
    }
  }, [predictionId, text, predictedLabel])
  const concepts = reading.id === predictionId ? reading.data : null

  if (!item) {
    return null
  }

  const shapValues = (tokenImportances || []).map(({ token, importance, position, text }) => ({
    token,
    value: importance,
    position,
    text,
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
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-6 py-5">
          <div>
            <p className="section-kicker">Explanation</p>
            <h3 className="mt-2 text-xl font-semibold text-slate-900">{item.predicted_label}</h3>
          </div>
          <button type="button" onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>

        <div className="space-y-6 p-6">
          {/* The passage itself first. The review card shows only an
              excerpt, and the explanation below takes several seconds — reading the
              text should not wait for it. Batch items and queue items both
              carry it as text_content. */}
          {item.text_content ? (
            <section className="dashboard-card">
              <p className="section-kicker">Text</p>
              <p className="mt-3 max-h-80 overflow-y-auto whitespace-pre-wrap text-base leading-7 text-slate-800">
                <ConceptText
                  text={item.text_content}
                  spans={concepts?.spans}
                  predictedLabel={item.predicted_label}
                />
              </p>
              {concepts?.spans?.length ? (
                <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
                  <span>Concept markers:</span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: 'rgb(16, 185, 129)' }} />
                    {item.predicted_label}, the model&apos;s label
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: 'rgb(245, 158, 11)' }} />
                    other labels (hover a word to see which)
                  </span>
                </p>
              ) : null}
              <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-slate-200 pt-4">
                <span className="text-sm font-medium text-slate-800">
                  Predicted: {item.predicted_label}
                </span>
                {item.confidence != null ? (
                  <>
                    <ConfidenceBadge confidence={item.confidence} />
                    <span className="font-mono text-sm text-slate-500">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </>
                ) : null}
              </div>
            </section>
          ) : null}

          {item.flag_reason === 'ontology_conflict' ? <OntologyFlag item={item} /> : null}

          <OntologyReading reading={concepts} predictedLabel={item.predicted_label} />

          {(status === 'requesting' || status === 'pending' || status === 'processing' || status === 'idle') && (
            <div className="dashboard-card flex flex-col items-center gap-4 py-12 text-center">
              <Spinner />
              <p className="text-sm font-medium text-slate-700">
                Generating explanation (usually under 15 s)...
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

          {status === 'done' && (
            <>
              {/* The chart first — the agreement card talks about the tokens
                  in it, so it reads as a conclusion rather than a verdict
                  handed down before the evidence. */}
              <ExplanationPanel shapValues={shapValues} />
              <AgreementCard agreement={agreement} />
            </>
          )}
        </div>
      </aside>
    </>
  )
}

export default ExplainPanel
