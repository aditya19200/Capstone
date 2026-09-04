import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getDashboardStats } from '../api/client.js'
import { useAuth } from '../auth/AuthContext.jsx'
import { CONFIDENCE_HIGH, REVIEW_THRESHOLD } from '../api/constants.js'
import ActivityList from '../components/dashboard/ActivityList.jsx'
import QuickActions from '../components/dashboard/QuickActions.jsx'
import StatsCard from '../components/dashboard/StatsCard.jsx'
import WorkflowSteps from '../components/dashboard/WorkflowSteps.jsx'

// Mirrors Sidebar.jsx's navigationByRole pattern — each role only sees
// shortcuts to routes it can actually reach, instead of one static list
// that silently bounces roles that don't match a given route.
const quickActionsByRole = {
  annotator: [{ label: 'Upload Dataset', to: '/annotate' }],
  reviewer: [{ label: 'Review Queue', to: '/review' }],
  admin: [
    { label: 'Manage Models', to: '/admin' },
    { label: 'View Metrics', to: '/metrics', variant: 'secondary' },
    { label: 'Trigger Retrain', to: '/retrain', variant: 'secondary' },
  ],
}

const pct = (part, whole) => (whole > 0 ? Math.round((part / whole) * 100) : 0)

// Turns an ISO timestamp into "10 mins ago" for the activity feed.
const relativeTime = (iso) => {
  if (!iso) return ''
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (Number.isNaN(seconds)) return ''
  if (seconds < 60) return 'just now'
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins} min${mins === 1 ? '' : 's'} ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`
  const days = Math.floor(hours / 24)
  return `${days} day${days === 1 ? '' : 's'} ago`
}

// Same routing vocabulary the backend's active-learning engine uses, derived
// from confidence so the activity feed's badges agree with the rest of the app.
const routingFor = (confidence) => {
  if (confidence >= CONFIDENCE_HIGH) return 'AUTO_ACCEPT'
  if (confidence < REVIEW_THRESHOLD) return 'ROUTE_TO_REVIEWER'
  return 'NEEDS_EXPLANATION'
}

function DashboardPage() {
  const navigate = useNavigate()
  const { role } = useAuth()
  const [data, setData] = useState(null) // null = loading
  const [fetchError, setFetchError] = useState('')

  const fetchStats = async () => {
    try {
      setData(await getDashboardStats())
      setFetchError('')
    } catch (error) {
      console.error('[Dashboard] getDashboardStats failed', error)
      setFetchError('Could not load dashboard statistics.')
    }
  }

  // Wrapped in an async IIFE, matching Review.jsx: there must be no
  // synchronous setState in the effect body, or react-hooks/set-state-in-effect
  // fires (a sync setState inside an effect forces an extra render).
  useEffect(() => {
    ;(async () => {
      await fetchStats()
    })()
  }, [])

  const quickActions = (quickActionsByRole[role] || quickActionsByRole.annotator).map((action) => ({
    ...action,
    onClick: () => navigate(action.to),
  }))

  if (data === null && !fetchError) {
    return (
      <section className="dashboard-card py-16 text-center text-sm text-slate-500">
        Loading dashboard...
      </section>
    )
  }

  if (fetchError) {
    return (
      <section className="dashboard-card border-dashed py-12 text-center">
        <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
          Something went wrong
        </p>
        <p className="mt-3 text-sm text-slate-600">{fetchError}</p>
        <button type="button" onClick={fetchStats} className="btn-primary mt-4 px-6 py-3">
          Retry
        </button>
      </section>
    )
  }

  const classified = data.documents_classified

  const stats = [
    {
      label: 'Total Documents',
      value: data.documents_uploaded.toLocaleString(),
      detail: `${classified.toLocaleString()} classified so far`,
      icon: 'DOC',
      accentClass: 'bg-indigo-100 text-indigo-700',
    },
    {
      label: 'Auto Accepted',
      value: data.auto_accepted.toLocaleString(),
      detail: `${pct(data.auto_accepted, classified)}% resolved without review`,
      icon: 'AUTO',
      accentClass: 'bg-green-100 text-green-700',
    },
    {
      label: 'Needs Review',
      value: data.needs_review.toLocaleString(),
      detail: 'Below the review confidence threshold',
      icon: 'REV',
      accentClass: 'bg-yellow-100 text-yellow-700',
    },
    {
      label: 'Flagged / Conflicts',
      value: data.conflicts.toLocaleString(),
      detail: 'Annotations flagged for disagreement',
      icon: 'FLAG',
      accentClass: 'bg-orange-100 text-orange-700',
    },
  ]

  const confidenceDistribution = [
    { range: 'High', count: data.confidence_high, color: '#4f46e5' },
    { range: 'Medium', count: data.confidence_medium, color: '#818cf8' },
    { range: 'Low', count: data.confidence_low, color: '#c7d2fe' },
  ]

  const workflowSteps = [
    {
      label: 'Uploaded',
      count: data.documents_uploaded.toLocaleString(),
      progress: data.documents_uploaded > 0 ? 100 : 0,
      description: 'Documents entered into the annotation workspace.',
    },
    {
      label: 'Predicted',
      count: classified.toLocaleString(),
      progress: pct(classified, data.documents_uploaded),
      description: 'Model inference completed for the current batch.',
    },
    {
      label: 'Reviewed',
      count: data.annotations_total.toLocaleString(),
      progress: pct(data.annotations_total, classified),
      description: 'Reviewer touchpoints on uncertain or routed cases.',
    },
    {
      label: 'Validated',
      count: data.annotations_validated.toLocaleString(),
      progress: pct(data.annotations_validated, classified),
      description: 'Human-confirmed labels available for retraining.',
    },
  ]

  const recentActivity = data.recent_activity.map((item) => ({
    id: item.prediction_id,
    documentName: item.text_excerpt,
    label: item.predicted_label,
    status: routingFor(item.confidence),
    timestamp: relativeTime(item.created_at),
  }))

  const topLabels = data.label_distribution.map((l) => ({
    label: l.label,
    percentage: l.percentage,
  }))

  const isEmpty = classified === 0

  return (
    <section className="space-y-6">
      <div className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Dashboard
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">Dashboard</h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Overview of annotation workflow and model performance
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat) => (
          <StatsCard key={stat.label} {...stat} />
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr),minmax(320px,1fr)]">
        <section className="dashboard-card">
          <div className="flex flex-col gap-3 border-b border-slate-200 pb-5 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="section-kicker">Confidence Distribution</p>
              <h3 className="mt-2 text-xl font-semibold text-slate-800">Prediction confidence</h3>
            </div>
            <p className="text-sm text-slate-500">
              {isEmpty
                ? 'No predictions yet.'
                : `${pct(data.confidence_high, classified)}% high confidence across ${classified.toLocaleString()} prediction${classified === 1 ? '' : 's'}.`}
            </p>
          </div>

          <div className="mt-6 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={confidenceDistribution} barSize={48}>
                <CartesianGrid vertical={false} stroke="#e2e8f0" />
                <XAxis
                  dataKey="range"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#64748b', fontSize: 12 }}
                />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  cursor={{ fill: '#f8fafc' }}
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 1px 2px 0 rgb(15 23 42 / 0.08)',
                  }}
                />
                <Bar dataKey="count" radius={[12, 12, 0, 0]}>
                  {confidenceDistribution.map((entry) => (
                    <Cell key={entry.range} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="dashboard-card">
          <p className="section-kicker">Quick Actions</p>
          <h3 className="mt-2 text-xl font-semibold text-slate-800">Move through the workflow</h3>
          <p className="mt-3 text-sm leading-6 text-slate-500">
            Jump directly into uploading, reviewing predictions, or exploring concept relationships.
          </p>
          <div className="mt-6">
            <QuickActions actions={quickActions} />
          </div>

          <div className="mt-8 border-t border-slate-200 pt-6">
            <p className="section-kicker">Most Predicted Categories</p>
            {topLabels.length === 0 ? (
              <p className="mt-4 text-sm text-slate-500">Nothing classified yet.</p>
            ) : null}
            <div className="mt-4 space-y-4">
              {topLabels.map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-slate-700">{item.label}</span>
                    <span className="text-slate-500">{item.percentage}%</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-indigo-600"
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>

      <section className="space-y-4">
        <div>
          <p className="section-kicker">Annotation Workflow Funnel</p>
          <h3 className="mt-2 text-xl font-semibold text-slate-800">Pipeline coverage</h3>
        </div>
        <WorkflowSteps steps={workflowSteps} />
      </section>

      <section className="dashboard-card">
        <div className="flex flex-col gap-3 border-b border-slate-200 pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="section-kicker">Recent Activity</p>
            <h3 className="mt-2 text-xl font-semibold text-slate-800">Latest workflow events</h3>
          </div>
          <p className="text-sm text-slate-500">
            {recentActivity.length > 0
              ? `Most recent ${recentActivity.length} prediction${recentActivity.length === 1 ? '' : 's'}.`
              : ''}
          </p>
        </div>

        <div className="mt-6">
          {recentActivity.length > 0 ? (
            <ActivityList items={recentActivity} />
          ) : (
            <p className="py-8 text-center text-sm text-slate-500">
              No activity yet — upload a batch on the Annotate screen to get started.
            </p>
          )}
        </div>
      </section>
    </section>
  )
}

export default DashboardPage
