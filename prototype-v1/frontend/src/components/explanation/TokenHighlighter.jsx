// The passage in reading order, each word shaded by how much it moved the
// model: red toward the predicted label, blue away from it.
//
// Shading is relative to the strongest word in this passage. Raw SHAP values
// here are small (a few hundredths of a probability), so fixed cut-offs left
// every word unshaded.
const tone = (value, max) => {
  const strength = max > 0 ? Math.abs(value) / max : 0
  if (strength < 0.15) return undefined
  const rgb = value > 0 ? '220, 38, 38' : '37, 99, 235'
  return { backgroundColor: `rgba(${rgb}, ${(0.12 + 0.5 * strength).toFixed(2)})` }
}

function TokenHighlighter({ tokens }) {
  // Entries carry their place in the text; older explanations, stored before
  // that was recorded, come in importance order and are shown as they are.
  const ordered = tokens.every((t) => t.position != null)
    ? [...tokens].sort((a, b) => a.position - b.position)
    : tokens
  const max = Math.max(0, ...tokens.map((t) => Math.abs(t.value)))

  return (
    <div className="max-h-80 overflow-y-auto rounded-2xl bg-slate-50 p-4">
      <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
        {ordered.map((item, index) => (
          <span
            key={`${item.token}-${index}`}
            className="rounded-sm"
            style={tone(item.value, max)}
            title={`${item.token}: ${item.value.toFixed(4)}`}
          >
            {item.text ?? `${item.token} `}
          </span>
        ))}
      </p>
    </div>
  )
}

export default TokenHighlighter
