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
import { useAuth } from '../auth/AuthContext.jsx'
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

const stats = [
  {
    label: 'Total Documents',
    value: '1,248',
    detail: 'Across current upload cycles',
    icon: 'DOC',
    accentClass: 'bg-indigo-100 text-indigo-700',
  },
  {
    label: 'Auto Accepted',
    value: '842',
    detail: '67.5% resolved without review',
    icon: 'AUTO',
    accentClass: 'bg-green-100 text-green-700',
  },
  {
    label: 'Needs Review',
    value: '286',
    detail: 'Queued for reviewer validation',
    icon: 'REV',
    accentClass: 'bg-yellow-100 text-yellow-700',
  },
  {
    label: 'Flagged / Conflicts',
    value: '120',
    detail: 'Cases requiring manual intervention',
    icon: 'FLAG',
    accentClass: 'bg-orange-100 text-orange-700',
  },
]

const confidenceDistribution = [
  { range: 'High', count: 624, color: '#4f46e5' },
  { range: 'Medium', count: 402, color: '#818cf8' },
  { range: 'Low', count: 222, color: '#c7d2fe' },
]

const workflowSteps = [
  {
    label: 'Uploaded',
    count: '1,248',
    progress: 100,
    description: 'Documents entered into the annotation workspace.',
  },
  {
    label: 'Predicted',
    count: '1,248',
    progress: 100,
    description: 'Model inference completed for the current batch.',
  },
  {
    label: 'Reviewed',
    count: '406',
    progress: 33,
    description: 'Reviewer touchpoints on uncertain or routed cases.',
  },
  {
    label: 'Accepted',
    count: '842',
    progress: 67,
    description: 'Predictions finalized and added to the workflow output.',
  },
]

const recentActivity = [
  {
    id: 'ACT-1001',
    documentName: 'Master Services Agreement - April',
    label: 'Contract Law',
    status: 'AUTO_ACCEPT',
    timestamp: '10 mins ago',
  },
  {
    id: 'ACT-1002',
    documentName: 'Constitutional Petition Draft',
    label: 'Constitutional Law',
    status: 'NEEDS_EXPLANATION',
    timestamp: '24 mins ago',
  },
  {
    id: 'ACT-1003',
    documentName: 'Employment Tribunal Filing',
    label: 'Labour & Employment Law',
    status: 'ROUTE_TO_REVIEWER',
    timestamp: '42 mins ago',
  },
  {
    id: 'ACT-1004',
    documentName: 'Property Lease Dispute Notice',
    label: 'Property Law',
    status: 'AUTO_ACCEPT',
    timestamp: '1 hour ago',
  },
]

const topLabels = [
  { label: 'Contract Law', percentage: 34 },
  { label: 'Constitutional Law', percentage: 27 },
  { label: 'Property Law', percentage: 18 },
  { label: 'Labour & Employment Law', percentage: 13 },
  { label: 'Criminal Law', percentage: 8 },
]

function DashboardPage() {
  const navigate = useNavigate()
  const { role } = useAuth()

  const quickActions = (quickActionsByRole[role] || quickActionsByRole.annotator).map((action) => ({
    ...action,
    onClick: () => navigate(action.to),
  }))

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
            <p className="text-sm text-slate-500">High-confidence predictions dominate this batch.</p>
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
            <p className="section-kicker">Model Insights</p>
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
          <p className="text-sm text-slate-500">Mock activity stream from the current batch.</p>
        </div>

        <div className="mt-6">
          <ActivityList items={recentActivity} />
        </div>
      </section>
    </section>
  )
}

export default DashboardPage
