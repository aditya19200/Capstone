function ModelVersionsTable({ versions, activatingId, onActivate }) {
  return (
    <div className="dashboard-card">
      <div className="border-b border-slate-200 pb-5">
        <p className="section-kicker">Model Versions</p>
        <h3 className="mt-2 text-xl font-semibold text-slate-900">Version history</h3>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-100">
            <tr>
              {['Version', 'Accuracy', 'Trained', 'Status', ''].map((heading) => (
                <th
                  key={heading}
                  scope="col"
                  className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.16em] text-slate-500"
                >
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 bg-white">
            {versions.map((version) => (
              <tr key={version.version_id}>
                <td className="px-4 py-4 text-sm font-medium text-slate-800">
                  {version.version_id}
                </td>
                <td className="px-4 py-4 text-sm text-slate-700">
                  {typeof version.accuracy === 'number' ? `${(version.accuracy * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="px-4 py-4 text-sm text-slate-600">
                  {new Date(version.trained_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-4">
                  {version.is_active ? (
                    <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-700">
                      Active
                    </span>
                  ) : (
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                      Inactive
                    </span>
                  )}
                </td>
                <td className="px-4 py-4 text-right">
                  <button
                    type="button"
                    className="btn-secondary"
                    disabled={version.is_active || activatingId === version.version_id}
                    onClick={() => onActivate(version)}
                  >
                    {activatingId === version.version_id ? 'Activating...' : 'Activate'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ModelVersionsTable
