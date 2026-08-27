function AnnotationActions({ onAction }) {
  const actions = [
    {
      label: 'Accept',
      value: 'accept',
      className: 'btn-primary',
    },
    {
      label: 'Modify',
      value: 'modify',
      className: 'btn-secondary',
    },
    {
      label: 'Flag Uncertain',
      value: 'flag',
      className: 'btn-danger',
    },
  ]

  return (
    <section className="dashboard-card">
      <div className="border-b border-slate-200 pb-5">
        <p className="section-kicker">Annotation Actions</p>
        <h3 className="mt-2 text-xl font-semibold text-slate-800">Resolve this prediction</h3>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {actions.map((action) => (
          <button
            key={action.value}
            type="button"
            onClick={() => onAction(action.value)}
            className={action.className}
          >
            {action.label}
          </button>
        ))}
      </div>
    </section>
  )
}

export default AnnotationActions
