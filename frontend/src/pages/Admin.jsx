import { useEffect, useState } from 'react'
import { activateModelVersion, getAdminMetrics } from '../api/client.js'
import ModelVersionsTable from '../components/admin/ModelVersionsTable.jsx'
import StatsCard from '../components/dashboard/StatsCard.jsx'

const STATUS_CARD_META = {
  pending: { label: 'Pending', accentClass: 'bg-yellow-100 text-yellow-700', icon: 'PEN' },
  validated: { label: 'Validated', accentClass: 'bg-green-100 text-green-700', icon: 'VAL' },
  rejected: { label: 'Rejected', accentClass: 'bg-red-100 text-red-700', icon: 'REJ' },
}

function AdminPage() {
  const [metrics, setMetrics] = useState(null) // null = loading
  const [fetchError, setFetchError] = useState('')
  const [activatingId, setActivatingId] = useState(null)
  const [activateError, setActivateError] = useState('')

  const fetchMetrics = async () => {
    try {
      const result = await getAdminMetrics()
      setMetrics(result)
      setFetchError('')
    } catch (error) {
      console.error('[Admin] getAdminMetrics failed', error)
      setFetchError('Could not load admin metrics. Please try again.')
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

  const handleActivate = async (version) => {
    if (activatingId) {
      return
    }

    setActivateError('')
    setActivatingId(version.version_id)

    const previousVersions = metrics.model_versions
    setMetrics((prev) => ({
      ...prev,
      model_versions: prev.model_versions.map((row) => ({
        ...row,
        is_active: row.version_id === version.version_id,
      })),
    }))

    try {
      await activateModelVersion(version.version_id)
      setActivatingId(null)
    } catch (error) {
      console.error('[Admin] activateModelVersion failed', error)
      setMetrics((prev) => ({ ...prev, model_versions: previousVersions }))
      setActivatingId(null)
      setActivateError(
        `Could not activate version "${version.version_id}" — reverted to the previous state.`,
      )
    }
  }

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 py-4">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Admin Workspace
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Model &amp; Annotation Administration
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Manage model versions and monitor annotation throughput.
        </p>
      </div>

      {activateError ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
          {activateError}
        </div>
      ) : null}

      {metrics === null && !fetchError ? (
        <div className="dashboard-card py-12 text-center text-sm text-slate-500">
          Loading admin metrics...
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
        <>
          <div className="grid gap-4 md:grid-cols-3">
            {Object.entries(STATUS_CARD_META).map(([status, meta]) => (
              <StatsCard
                key={status}
                label={`${meta.label} Annotations`}
                value={metrics.annotation_counts_by_status[status] ?? 0}
                accentClass={meta.accentClass}
                icon={meta.icon}
              />
            ))}
          </div>

          <ModelVersionsTable
            versions={metrics.model_versions}
            activatingId={activatingId}
            onActivate={handleActivate}
          />
        </>
      )}
    </section>
  )
}

export default AdminPage
