const getTokenTone = (value) => {
  const magnitude = Math.abs(value)

  if (value > 0) {
    if (magnitude > 0.75) return 'bg-red-200 text-red-900 ring-1 ring-red-300'
    if (magnitude > 0.4) return 'bg-orange-100 text-orange-900 ring-1 ring-orange-200'
    if (magnitude > 0.15) return 'bg-amber-50 text-amber-900 ring-1 ring-amber-100'
  }

  if (value < 0) {
    if (magnitude > 0.75) return 'bg-blue-200 text-blue-900 ring-1 ring-blue-300'
    if (magnitude > 0.4) return 'bg-sky-100 text-sky-900 ring-1 ring-sky-200'
    if (magnitude > 0.15) return 'bg-cyan-50 text-cyan-900 ring-1 ring-cyan-100'
  }

  return 'text-slate-700'
}

function TokenHighlighter({ tokens }) {
  return (
    <div className="overflow-hidden rounded-2xl bg-slate-50 p-4">
      <div className="flex flex-wrap gap-2">
      {tokens.map((item, index) => (
        <span
          key={`${item.token}-${index}`}
          className={[
            'max-w-full break-words rounded-md px-2 py-1 text-sm leading-6 transition',
            getTokenTone(item.value),
          ].join(' ')}
        >
          {item.token}
        </span>
      ))}
      </div>
    </div>
  )
}

export default TokenHighlighter
