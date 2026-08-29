import { useState } from 'react'
import { LEGAL_LABELS } from '../../api/constants.js'
import ConfidenceBadge from '../prediction/ConfidenceBadge.jsx'

function ReviewQueueItem({ item, isSubmitting, onSubmit }) {
  const [selectedLabel, setSelectedLabel] = useState(item.predicted_label)

  return (
    <div className="dashboard-card">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          <p className="text-sm text-slate-700">{item.text_preview}</p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span className="text-sm font-medium text-slate-800">
              Predicted: {item.predicted_label}
            </span>
            <ConfidenceBadge confidence={item.confidence} />
          </div>
        </div>

        <div className="flex w-full flex-col gap-3 sm:w-64">
          <select
            value={selectedLabel}
            onChange={(event) => setSelectedLabel(event.target.value)}
            disabled={isSubmitting}
          >
            {LEGAL_LABELS.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>

          <button
            type="button"
            className="btn-primary"
            disabled={isSubmitting}
            onClick={() => onSubmit(item, selectedLabel)}
          >
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ReviewQueueItem
