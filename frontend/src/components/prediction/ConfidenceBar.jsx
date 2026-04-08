function ConfidenceBar({ confidenceScore }) {
  const percentage = Math.round((confidenceScore || 0) * 100)
  const totalSegments = 20
  const filledSegments = Math.round((percentage / 100) * totalSegments)

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          Confidence Score
        </p>
        <span className="text-sm font-semibold text-slate-900">{percentage}%</span>
      </div>

      <div className="grid grid-cols-[repeat(20,minmax(0,1fr))] gap-1">
        {Array.from({ length: totalSegments }, (_, index) => (
          <span
            key={index}
            className={[
              'h-3 rounded-full transition-colors',
              index < filledSegments ? 'bg-indigo-600' : 'bg-indigo-100',
            ].join(' ')}
          />
        ))}
      </div>
    </div>
  )
}

export default ConfidenceBar
