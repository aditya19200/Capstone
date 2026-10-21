import { useEffect, useState } from 'react'
import { downloadDataset, downloadDatasetPdf, getDatasetCard } from '../../api/client.js'

// The output of this system is the annotated corpus, not the classifier.
// This is where that stops being a claim: it downloads the validated
// annotations in the shape the retrain worker consumes, so the file someone
// takes away is literally the thing we retrain on.
//
// Two formats, because they answer two different questions. JSONL is what
// another system ingests — one object per line, straight into
// datasets.load_dataset. The PDF is what a person reads: each document with
// its ontology concept markers highlighted in place, the model's prediction
// and confidence, and the human's decision beside it. Same records, same
// rows() call, so they cannot describe different corpora.

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

  const handleDownload = async (format) => {
    setBusy(format)
    try {
      const { blob, filename } =
        format === 'pdf' ? await downloadDatasetPdf() : await downloadDataset()
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
      // responseType 'blob' turns an error body into a Blob as well, so the
      // detail has to be read out of it rather than off response.data.
      let detail = err?.response?.data?.detail
      if (err?.response?.data instanceof Blob) {
        try {
          detail = JSON.parse(await err.response.data.text())?.detail
        } catch {
          detail = undefined
        }
      }
      setError(detail || 'Download failed.')
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
            Every human-validated annotation. This is the same data the retrain
            worker consumes — the export and the training set are assembled from
            one place so they cannot drift apart.
          </p>
          <p className="mt-2 max-w-xl text-sm leading-6 text-slate-500">
            <span className="font-medium text-slate-700">.jsonl</span> for
            another system to ingest.{' '}
            <span className="font-medium text-slate-700">PDF</span> to read: each
            document with its concept markers highlighted, what the model
            predicted, and what the human decided.
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          <button
            type="button"
            className="btn-primary px-6 py-3"
            onClick={() => handleDownload('pdf')}
            disabled={Boolean(busy) || empty}
          >
            {busy === 'pdf' ? 'Rendering...' : 'Download PDF'}
          </button>
          <button
            type="button"
            className="btn-secondary px-6 py-3"
            onClick={() => handleDownload('jsonl')}
            disabled={Boolean(busy) || empty}
          >
            {busy === 'jsonl' ? 'Preparing...' : 'Download .jsonl'}
          </button>
        </div>
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
