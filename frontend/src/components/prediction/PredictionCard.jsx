import ConfidenceBar from './ConfidenceBar.jsx'
import StatusBadge from './StatusBadge.jsx'

function PredictionCard({ predictedLabel, confidenceScore, routingDecision }) {
  return (
    <div className="dashboard-card">
      <div className="flex flex-col gap-3 border-b border-slate-200 pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="section-kicker">Prediction Result</p>
          <h3 className="mt-2 text-xl font-semibold text-slate-800">Model Output Preview</h3>
        </div>
        <StatusBadge routingDecision={routingDecision} />
      </div>

      <div className="mt-6 space-y-5">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            Predicted Label
          </p>
          <p className="mt-2 text-lg font-semibold text-slate-800">{predictedLabel}</p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <ConfidenceBar confidenceScore={confidenceScore} />
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            Routing Decision
          </p>
          <div className="mt-2">
            <StatusBadge routingDecision={routingDecision} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default PredictionCard
