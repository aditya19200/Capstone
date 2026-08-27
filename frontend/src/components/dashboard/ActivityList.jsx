import StatusBadge from '../prediction/StatusBadge.jsx'

function ActivityList({ items }) {
  return (
    <div className="space-y-4">
      {items.map((item) => (
        <article
          key={item.id}
          className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 md:flex-row md:items-center md:justify-between"
        >
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-800">{item.documentName}</p>
            <p className="mt-1 text-sm text-slate-500">{item.label}</p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <StatusBadge routingDecision={item.status} />
            <span className="text-sm text-slate-500">{item.timestamp}</span>
          </div>
        </article>
      ))}
    </div>
  )
}

export default ActivityList
