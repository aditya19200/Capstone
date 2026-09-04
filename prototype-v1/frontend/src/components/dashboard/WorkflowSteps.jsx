function WorkflowSteps({ steps }) {
  return (
    <div className="grid gap-4 lg:grid-cols-4">
      {steps.map((step, index) => (
        <article key={step.label} className="relative">
          <div className="dashboard-card h-full">
            <div className="flex items-center justify-between">
              <span className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-indigo-700">
                Stage {index + 1}
              </span>
              <span className="text-sm font-medium text-slate-500">{step.count}</span>
            </div>
            <h4 className="mt-4 text-lg font-semibold text-slate-800">{step.label}</h4>
            <p className="mt-2 text-sm text-slate-500">{step.description}</p>
            <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full rounded-full bg-indigo-600"
                style={{ width: `${step.progress}%` }}
              />
            </div>
          </div>

          {index < steps.length - 1 ? (
            <div className="absolute right-[-10px] top-1/2 hidden h-px w-5 -translate-y-1/2 bg-slate-300 lg:block" />
          ) : null}
        </article>
      ))}
    </div>
  )
}

export default WorkflowSteps
