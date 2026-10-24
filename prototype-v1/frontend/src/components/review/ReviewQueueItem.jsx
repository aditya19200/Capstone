import { useState } from 'react'
import useLabels from '../../hooks/useLabels.js'
import ConfidenceBadge from '../prediction/ConfidenceBadge.jsx'
import OntologyFlag from './OntologyFlag.jsx'

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
        <div className="mb-4">
          <OntologyFlag item={item} />
        </div>
      ) : null}

      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          <p className="text-base leading-7 text-slate-800">{item.text_preview}</p>
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
            {item.superseded ? (
              <span
                className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600"
                title="This prediction was made before the current version was activated. Re-upload the document to see what the serving model says."
              >
                from {item.predicted_by}, superseded
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
            {/* Opens the drawer: the whole passage (the card shows only the
                first 200 characters), the ontology's reading, and which words
                drove the model's choice — judging a call the model was unsure
                about is hard from an excerpt and a label alone. */}
            <button
              type="button"
              className="btn-secondary flex-1"
              disabled={isSubmitting}
              onClick={() => onExplain(item)}
            >
              Read &amp; explain
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
