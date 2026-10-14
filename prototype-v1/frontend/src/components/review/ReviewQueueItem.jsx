import { useState } from 'react'
import useLabels from '../../hooks/useLabels.js'
import ConfidenceBadge from '../prediction/ConfidenceBadge.jsx'

function ReviewQueueItem({ item, isSubmitting, onSubmit, onExplain }) {
  const { labels } = useLabels()
  const isOntologyFlag = item.flag_reason === 'ontology_conflict'

  // Pre-select the ontology's suggestion when it raised the flag. The reviewer
  // still has to press Submit, and the banner says plainly that this is a
  // suggestion — but starting from the model's label would be actively
  // unhelpful here, since that label is the thing under question.
  const [selectedLabel, setSelectedLabel] = useState(
    isOntologyFlag && item.suggested_label ? item.suggested_label : item.predicted_label,
  )

  return (
    <div className="dashboard-card">
      {/* A confident prediction appearing in a review queue looks like a bug
          unless the screen says why it is here. */}
      {isOntologyFlag ? (
        <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-amber-600 dark:text-amber-300">
              Ontology conflict
            </span>
            <span className="text-sm text-slate-700 dark:text-slate-200">
              The model was confident, but the legal concepts in this text point
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
      ) : null}

      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          <p className="text-sm text-slate-700">{item.text_preview}</p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span className="text-sm font-medium text-slate-800">
              Predicted: {item.predicted_label}
            </span>
            <ConfidenceBadge confidence={item.confidence} />
            {isOntologyFlag ? (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                queued on concept evidence, not on confidence
              </span>
            ) : null}
          </div>
        </div>

        <div className="flex w-full flex-col gap-3 sm:w-64">
          <label className="sr-only" htmlFor={`review-label-${item.prediction_id}`}>
            Select final label
          </label>
          <select
            id={`review-label-${item.prediction_id}`}
            value={selectedLabel}
            onChange={(event) => setSelectedLabel(event.target.value)}
            disabled={isSubmitting}
          >
            {labels.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>

          <div className="flex gap-2">
            {/* Lets the reviewer see which words drove the model's choice
                before accepting or overriding it — the whole point of the
                low-confidence queue is judging a call the model wasn't sure
                about, which is hard to do from the label alone. */}
            <button
              type="button"
              className="btn-secondary flex-1"
              disabled={isSubmitting}
              onClick={() => onExplain(item)}
            >
              Explain
            </button>
            <button
              type="button"
              className="btn-primary flex-1"
              disabled={isSubmitting}
              onClick={() => onSubmit(item, selectedLabel)}
            >
              {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ReviewQueueItem
