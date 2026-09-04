import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) {
    return null
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-xl">
      <p className="text-sm font-semibold text-slate-900">{label}</p>
      <p className="mt-1 text-sm text-slate-600">SHAP value: {payload[0].value}</p>
    </div>
  )
}

function ShapBarChart({ tokens }) {
  const chartData = [...tokens]
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 8)
    .map((item) => ({
      token: item.token,
      value: item.value,
      fill: item.value >= 0 ? '#dc2626' : '#2563eb',
    }))
    .reverse()

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 12, left: 12, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fill: '#64748b', fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="token"
            tick={{ fill: '#334155', fontSize: 12 }}
            width={90}
          />
          <Tooltip cursor={{ fill: '#f8fafc' }} content={<CustomTooltip />} />
          <Bar dataKey="value" radius={[0, 6, 6, 0]}>
            {chartData.map((entry) => (
              <Cell key={entry.token} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default ShapBarChart
