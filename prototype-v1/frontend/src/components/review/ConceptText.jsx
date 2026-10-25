// The passage with the ontology's concept markers marked where they occur, so
// a reviewer can see which words each label's evidence rests on. Green marks
// the model's own label; amber, any other label's.
const TONE = {
  own: {
    backgroundColor: 'rgba(16, 185, 129, 0.18)',
    boxShadow: 'inset 0 -2px 0 rgba(16, 185, 129, 0.85)',
    color: 'inherit',
  },
  other: {
    backgroundColor: 'rgba(245, 158, 11, 0.2)',
    boxShadow: 'inset 0 -2px 0 rgba(245, 158, 11, 0.9)',
    color: 'inherit',
  },
}

// The backend counts offsets in code points and JavaScript in UTF-16 units;
// they agree unless the text holds a character outside the BMP (an emoji, say).
const ASTRAL = /[\uD800-\uDFFF]/

function ConceptText({ text, spans, predictedLabel }) {
  if (!spans?.length || ASTRAL.test(text)) return text

  const parts = []
  let at = 0
  spans.forEach((span, index) => {
    if (span.start > at) parts.push(text.slice(at, span.start))
    parts.push(
      <mark
        key={index}
        className="rounded-sm"
        style={span.label === predictedLabel ? TONE.own : TONE.other}
        title={`${span.label} concept marker`}
      >
        {text.slice(span.start, span.end)}
      </mark>,
    )
    at = span.end
  })
  if (at < text.length) parts.push(text.slice(at))
  return <>{parts}</>
}

export default ConceptText
