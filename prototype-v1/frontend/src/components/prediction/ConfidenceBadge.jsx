import { REVIEW_THRESHOLD } from '../../api/constants.js'

// Thresholds per BIDISHA_TASKS.md Task 2 spec: green >= 0.7, yellow 0.5-0.7,
// red < 0.5. The red/yellow boundary is REVIEW_THRESHOLD from the shared
// constant (mirrors the backend) — never hardcode a different number there.
const HIGH_THRESHOLD = 0.7

const bandFor = (confidence) => {
  if (confidence >= HIGH_THRESHOLD) {
    return { label: 'High', className: 'bg-green-100 text-green-700' }
  }
  if (confidence >= REVIEW_THRESHOLD) {
    return { label: 'Medium', className: 'bg-yellow-100 text-yellow-700' }
  }
  return { label: 'Low', className: 'bg-red-100 text-red-700' }
}

function ConfidenceBadge({ confidence }) {
  const { label, className } = bandFor(confidence ?? 0)

  return (
    <span
      className={[
        'inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em]',
        className,
      ].join(' ')}
    >
      Model certainty: {label}
    </span>
  )
}

export default ConfidenceBadge
