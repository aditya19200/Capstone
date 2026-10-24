import { useEffect, useState } from 'react'
import { getComparableBatches, compareBatches } from '../api/client.js'

// Shortens "b3f9c2a1-..." to something a person can match against the picker
// without reading 36 characters of uuid.
const shortId = (id) => (id ? `${id.slice(0, 8)}` : '')

const fmtTime = (iso) => {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString()
}

function ComparePage() {
  const [batches, setBatches] = useState(null)
  const [batchA, setBatchA] = useState('')
  const [batchB, setBatchB] = useState('')
  const [result, setResult] = useState(null)
  const [changedOnly, setChangedOnly] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    ;(async () => {
      try {
        const rows = await getComparableBatches()
        setBatches(rows)
        // Newest is first from the API. Default to comparing the two most
        // recent runs, which is almost always what you opened this for.
        if (rows.length >= 2) {
          setBatchA(rows[1].batch_id)
          setBatchB(rows[0].batch_id)
        }
      } catch (err) {
        console.error('[Compare] could not load batches', err)
        setError('Could not load the list of batches.')
        setBatches([])
      }
    })()
  }, [])

  const runCompare = async (onlyChanged = changedOnly) => {
    if (!batchA || !batchB) return
    setBusy(true)
    try {
      setResult(await compareBatches(batchA, batchB, onlyChanged))
      setError('')
    } catch (err) {
      console.error('[Compare] comparison failed', err)
      setError(
        err?.response?.data?.detail || 'Comparison failed. Pick two different batches.',
      )
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  const option = (b) =>
    `${b.filename || shortId(b.batch_id)} · ${b.item_count} docs · ${b.model_versions.join(', ')} · ${fmtTime(b.created_at)}`

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 py-4">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Admin Workspace
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Compare Two Runs
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Upload the same documents before and after a retrain, then line the two runs up
          here. Rows are matched on their text, so only the model differs.
        </p>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
          {error}
        </div>
      ) : null}

      <section className="dashboard-card">
        {batches === null ? (
          <p className="py-6 text-center text-sm text-slate-500">Loading batches...</p>
        ) : batches.length < 2 ? (
          <p className="py-6 text-center text-sm text-slate-500">
            Needs at least two batches. Upload the same file again after retraining.
          </p>
        ) : (
          <div className="grid gap-4 md:grid-cols-[1fr,1fr,auto] md:items-end">
            <div>
              <label className="section-kicker" htmlFor="batch-a">
                Before
              </label>
              <select
                id="batch-a"
                className="mt-2 w-full"
                value={batchA}
                onChange={(e) => setBatchA(e.target.value)}
              >
                {batches.map((b) => (
                  <option key={b.batch_id} value={b.batch_id}>
                    {option(b)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="section-kicker" htmlFor="batch-b">
                After
              </label>
              <select
                id="batch-b"
                className="mt-2 w-full"
                value={batchB}
                onChange={(e) => setBatchB(e.target.value)}
              >
                {batches.map((b) => (
                  <option key={b.batch_id} value={b.batch_id}>
                    {option(b)}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="button"
              className="btn-primary px-6 py-3"
              disabled={busy}
              onClick={() => runCompare()}
            >
              {busy ? 'Comparing...' : 'Compare'}
            </button>
          </div>
        )}
      </section>

      {result ? (
        <>
          <section className="dashboard-card">
            <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
              <div>
                <p className="section-kicker">Documents matched</p>
                <p className="mt-1 text-3xl font-semibold tabular-nums text-slate-800">
                  {result.matched}
                </p>
              </div>
              <div>
                <p className="section-kicker">Labels changed</p>
                <p className="mt-1 text-3xl font-semibold tabular-nums text-indigo-600">
                  {result.changed}
                </p>
              </div>
              <div className="min-w-0">
                <p className="section-kicker">Models</p>
                <p className="mt-1 font-mono text-sm text-slate-600">
                  {result.model_version_a} → {result.model_version_b}
                </p>
              </div>
              <label className="ml-auto flex items-center gap-2 text-sm text-slate-600">
                <input
                  type="checkbox"
                  checked={changedOnly}
                  onChange={(e) => {
                    setChangedOnly(e.target.checked)
                    runCompare(e.target.checked)
                  }}
                />
                Changed only
              </label>
            </div>
          </section>

          <section className="dashboard-card overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left">
                  <th className="pb-3 pr-4 font-semibold text-slate-500">Document</th>
                  <th className="pb-3 pr-4 font-semibold text-slate-500">Before</th>
                  <th className="pb-3 pr-4 font-semibold text-slate-500">After</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr
                    key={row.text_excerpt}
                    className={`border-b border-slate-100 align-top ${
                      row.changed ? 'bg-indigo-500/5' : ''
                    }`}
                  >
                    <td className="max-w-md py-3 pr-4 text-slate-700">{row.text_excerpt}</td>
                    <td className="py-3 pr-4 whitespace-nowrap">
                      <span className={row.changed ? 'text-slate-500 line-through' : 'text-slate-700'}>
                        {row.label_a}
                      </span>
                      <span className="ml-2 font-mono text-xs text-slate-400 tabular-nums">
                        {row.confidence_a.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-3 pr-4 whitespace-nowrap">
                      <span
                        className={
                          row.changed ? 'font-semibold text-indigo-600' : 'text-slate-700'
                        }
                      >
                        {row.label_b}
                      </span>
                      <span className="ml-2 font-mono text-xs text-slate-400 tabular-nums">
                        {row.confidence_b.toFixed(3)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {result.rows.length === 0 ? (
              <p className="py-8 text-center text-sm text-slate-500">
                No rows to show. If "Changed only" is ticked, nothing changed between these
                two runs.
              </p>
            ) : null}
          </section>
        </>
      ) : null}
    </section>
  )
}

export default ComparePage
