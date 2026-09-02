function StatsCard({ label, value, detail, accentClass = 'bg-indigo-100 text-indigo-700', icon }) {
  return (
    <article className="dashboard-card">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">{label}</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight text-slate-800">{value}</p>
          {detail ? <p className="mt-2 text-sm text-slate-500">{detail}</p> : null}
        </div>
        <div className={`rounded-xl px-3 py-2 text-sm font-semibold ${accentClass}`}>{icon}</div>
      </div>
    </article>
  )
}

export default StatsCard
