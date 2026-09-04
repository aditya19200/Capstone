function BatchProgress({ completedItems, totalItems, note }) {
  const percentage = totalItems > 0 ? Math.round((completedItems / totalItems) * 100) : 0

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between">
        <p className="section-kicker">Batch Progress</p>
        <span className="text-sm font-semibold text-slate-700">
          {completedItems} / {totalItems} items
        </span>
      </div>

      <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full bg-indigo-600 transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>

      <p className="mt-2 text-sm text-slate-500">
        {note || `${percentage}% complete — classifying documents...`}
      </p>
    </div>
  )
}

export default BatchProgress
