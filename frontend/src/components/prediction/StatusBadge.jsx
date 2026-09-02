const badgeStyles = {
  AUTO_ACCEPT: 'bg-green-100 text-green-700',
  NEEDS_EXPLANATION: 'bg-yellow-100 text-yellow-700',
  ROUTE_TO_REVIEWER: 'bg-orange-100 text-orange-700',
}

function StatusBadge({ routingDecision }) {
  return (
    <span
      className={[
        'inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]',
        badgeStyles[routingDecision] || 'bg-slate-100 text-slate-700',
      ].join(' ')}
    >
      {routingDecision?.replaceAll('_', ' ') || 'Unknown'}
    </span>
  )
}

export default StatusBadge
