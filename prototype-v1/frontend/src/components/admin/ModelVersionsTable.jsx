// Two different facts live in this table and they are not the same fact.
//
//   is_active   the database's record of which weights *should* be loaded.
//               Set by the Activate button, read at startup.
//   is_serving  which weights the running API actually has in memory, read
//               from the model service itself.
//
// They agree almost always. The moments they do not are the ones worth
// seeing: between pressing Activate and the restart that honours it, and
// after a startup that could not read a flagged version's files and fell
// back. A table showing only is_active reports a model that is answering
// nothing — which is exactly how the baseline came to show "Inactive" while
// serving every prediction on the screen behind it.
function StatusBadges({ version }) {
  const badge = 'rounded-full px-3 py-1 text-xs font-semibold'
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {version.is_serving ? (
        <span className={`${badge} bg-green-100 text-green-700`}>Serving</span>
      ) : null}
      {version.is_active ? (
        <span
          className={
            version.is_serving
              ? `${badge} bg-green-100 text-green-700`
              : `${badge} bg-amber-100 text-amber-700`
          }
          title={
            version.is_serving
              ? 'Flagged active in the database, and loaded.'
              : 'Flagged active in the database, but not loaded — restart the API to serve it.'
          }
        >
          {version.is_serving ? 'Active' : 'Active · restart pending'}
        </span>
      ) : null}
      {!version.is_serving && !version.is_active ? (
        <span className={`${badge} bg-slate-100 text-slate-600`}>Inactive</span>
      ) : null}
    </div>
  )
}

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
                  {/* version_number is the readable label ("v-20260903-093923");
                      version_id is the uuid the activate endpoint matches on. */}
                  {version.version_number ?? version.version_id}
                </td>
                <td className="px-4 py-4 text-sm text-slate-700">
                  {typeof version.accuracy === 'number' ? `${(version.accuracy * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="px-4 py-4 text-sm text-slate-600">
                  {new Date(version.trained_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-4">
                  <StatusBadges version={version} />
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

      <p className="mt-4 text-xs leading-5 text-slate-500">
        <span className="font-semibold text-slate-600">Serving</span> — these
        weights are loaded in the running API and are answering predictions
        right now.{' '}
        <span className="font-semibold text-slate-600">Active</span> — the
        database&apos;s record of which version should load. Activating sets the
        flag and loads the weights in one step, so the two normally match.
      </p>
    </div>
  )
}

export default ModelVersionsTable
