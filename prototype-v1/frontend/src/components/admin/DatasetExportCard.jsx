import { useEffect, useState } from 'react'
import { downloadDataset, getDatasetCard } from '../../api/client.js'

// The output of this system is the annotated corpus, not the classifier.
// This is where that stops being a claim: it downloads the validated
// annotations as JSONL, in the shape the retrain worker consumes, so the
// file someone takes away is literally the thing we retrain on.

function DatasetExportCard() {
  const [stats, setStats] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await getDatasetCard()
        if (!cancelled) setStats(data.stats)
      } catch (err) {
        console.error('[DatasetExport] could not load card', err)
        if (!cancelled) setError('Could not load dataset stats.')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const handleDownload = async () => {
    setBusy(true)
    try {
      const { blob, filename } = await downloadDataset()
      // Object URL rather than a direct link: the endpoint is admin-only and
      // a browser navigation would not carry the role header.
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      setError('')
    } catch (err) {
      console.error('[DatasetExport] download failed', err)
      setError(err?.response?.data?.detail || 'Download failed.')
    } finally {
      setBusy(false)
    }
  }

  const empty = stats && stats.total === 0

  return (
    <section className="dashboard-card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="section-kicker">Training data</p>
          <h3 className="mt-2 text-lg font-semibold text-slate-900">
            Validated corpus
          </h3>
          <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
            Every human-validated annotation, as JSON Lines. This is the same
            data the retrain worker consumes — the export and the training set
            are assembled from one place so they cannot drift apart.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary shrink-0 px-6 py-3"
          onClick={handleDownload}
          disabled={busy || empty}
        >
          {busy ? 'Preparing...' : 'Download .jsonl'}
        </button>
      </div>

      {stats ? (
        <div className="mt-5 flex flex-wrap gap-x-8 gap-y-3">
          <div>
            <p className="section-kicker">Examples</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-800">
              {stats.total}
            </p>
          </div>
          <div>
            <p className="section-kicker">Corrected by a human</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-indigo-600">
              {stats.corrected}
            </p>
          </div>
          <div>
            <p className="section-kicker">Confirmed as predicted</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-800">
              {stats.confirmed}
            </p>
          </div>
          {stats.labels_unused?.length ? (
            <div className="min-w-0">
              <p className="section-kicker">Labels with no examples</p>
              <p className="mt-1 text-sm text-slate-600">
                {stats.labels_unused.length} of{' '}
                {Object.keys(stats.label_counts || {}).length}
              </p>
            </div>
          ) : null}
        </div>
      ) : null}

      {empty ? (
        <p className="mt-4 text-sm text-slate-500">
          Nothing to export yet — validate some annotations in the Review queue
          first.
        </p>
      ) : null}

      {error ? (
        <p className="mt-4 text-sm font-medium text-red-700">{error}</p>
      ) : null}
    </section>
  )
}

export default DatasetExportCard
