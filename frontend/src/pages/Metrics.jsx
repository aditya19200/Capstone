import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getAdminMetrics } from '../api/client.js'

function MetricsPage() {
  const [metrics, setMetrics] = useState(null) // null = loading
  const [fetchError, setFetchError] = useState('')

  const fetchMetrics = async () => {
    try {
      const result = await getAdminMetrics()
      setMetrics(result)
      setFetchError('')
    } catch (error) {
      console.error('[Metrics] getAdminMetrics failed', error)
      setFetchError('Could not load model metrics. Please try again.')
      setMetrics(null)
    }
  }

  useEffect(() => {
    ;(async () => {
      await fetchMetrics()
    })()
  }, [])

  const handleRetry = () => {
    setMetrics(null)
    setFetchError('')
    fetchMetrics()
  }

  const chartData = metrics
    ? Object.entries(metrics.f1_per_class).map(([label, score]) => ({ label, score }))
    : []

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 py-4">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Model Metrics
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Per-class performance
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          F1 score for each legal category, from the active model version.
        </p>
      </div>

      {metrics === null && !fetchError ? (
        <div className="dashboard-card py-12 text-center text-sm text-slate-500">
          Loading model metrics...
        </div>
      ) : fetchError ? (
        <div className="dashboard-card border-dashed text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
            Something went wrong
          </p>
          <p className="mt-3 text-sm text-slate-600">{fetchError}</p>
          <button type="button" onClick={handleRetry} className="btn-primary mt-4 px-6 py-3">
            Retry
          </button>
        </div>
      ) : (
        <div className="dashboard-card">
          <div className="border-b border-slate-200 pb-5">
            <p className="section-kicker">Per-Class F1</p>
            <h3 className="mt-2 text-xl font-semibold text-slate-800">F1 score by legal category</h3>
          </div>

          <div className="mt-6 h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ bottom: 60 }}>
                <CartesianGrid vertical={false} stroke="#e2e8f0" />
                <XAxis
                  dataKey="label"
                  axisLine={false}
                  tickLine={false}
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                  tick={{ fill: '#64748b', fontSize: 11 }}
                />
                <YAxis
                  domain={[0, 1]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#64748b', fontSize: 12 }}
                />
                <Tooltip
                  cursor={{ fill: '#f8fafc' }}
                  formatter={(value) => value.toFixed(3)}
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 1px 2px 0 rgb(15 23 42 / 0.08)',
                  }}
                />
                <Bar dataKey="score" fill="#4f46e5" radius={[12, 12, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </section>
  )
}

export default MetricsPage
