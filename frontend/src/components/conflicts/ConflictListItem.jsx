import { useState } from 'react'
import { LEGAL_LABELS } from '../../api/constants.js'

function formatWhen(isoString) {
  return new Date(isoString).toLocaleString()
}

function ConflictListItem({ item, isSubmitting, onResolve }) {
  const [selectedLabel, setSelectedLabel] = useState(item.second_annotation.final_label)

  return (
    <div className="dashboard-card">
      <p className="text-sm text-slate-700">{item.text_preview}</p>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            AI Predicted
          </p>
          <p className="mt-2 text-sm font-medium text-slate-800">{item.predicted_label}</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {item.user_id}
          </p>
          <p className="mt-2 text-sm font-medium text-slate-800">{item.final_label}</p>
          <p className="mt-1 text-xs text-slate-500">{formatWhen(item.annotated_at)}</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {item.second_annotation.user_id}
          </p>
          <p className="mt-2 text-sm font-medium text-slate-800">
            {item.second_annotation.final_label}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {formatWhen(item.second_annotation.annotated_at)}
          </p>
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 border-t border-slate-200 pt-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <label
            className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500"
            htmlFor={`conflict-label-${item.annotation_id}`}
          >
            Resolve with final label
          </label>
          <select
            id={`conflict-label-${item.annotation_id}`}
            value={selectedLabel}
            onChange={(event) => setSelectedLabel(event.target.value)}
            disabled={isSubmitting}
            className="mt-2 block w-full sm:w-64"
          >
            {LEGAL_LABELS.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          className="btn-primary"
          disabled={isSubmitting}
          onClick={() => onResolve(item, selectedLabel)}
        >
          {isSubmitting ? 'Resolving...' : 'Resolve'}
        </button>
      </div>
    </div>
  )
}

export default ConflictListItem
