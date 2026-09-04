import { useEffect, useState } from 'react'
import { getConflicts, submitAnnotation } from '../api/client.js'
import ConflictListItem from '../components/conflicts/ConflictListItem.jsx'

function ConflictsPage() {
  const [items, setItems] = useState(null) // null = loading
  const [fetchError, setFetchError] = useState('')
  const [resolveError, setResolveError] = useState('')
  const [submittingIds, setSubmittingIds] = useState(() => new Set())

  const fetchConflicts = async () => {
    try {
      const conflicts = await getConflicts()
      setItems(conflicts)
      setFetchError('')
    } catch (error) {
      console.error('[Conflicts] failed to load conflicts', error)
      setFetchError('Could not load conflicts. Please try again.')
      setItems([])
    }
  }

  useEffect(() => {
    ;(async () => {
      await fetchConflicts()
    })()
  }, [])

  const handleRetry = () => {
    setItems(null)
    setFetchError('')
    fetchConflicts()
  }

  // Same race-proof double-submit guard as Review.jsx's handleSubmit — see
  // its comments for why the check has to happen inside the functional
  // updater rather than against the outer submittingIds closure.
  const handleResolve = async (item, selectedLabel) => {
    let alreadySubmitting = false
    setSubmittingIds((prev) => {
      if (prev.has(item.annotation_id)) {
        alreadySubmitting = true
        return prev
      }
      return new Set(prev).add(item.annotation_id)
    })

    if (alreadySubmitting) {
      return
    }

    setResolveError('')
    setItems((prev) => prev.filter((row) => row.annotation_id !== item.annotation_id))

    const action = selectedLabel === item.predicted_label ? 'accept' : 'modify'

    try {
      // This submits a third, authoritative annotation — the backend has no
      // "mark conflict resolved" endpoint, so has_conflict on the two
      // original rows never actually flips back to false. This is the best
      // available action today, not a literal resolution on its own record.
      await submitAnnotation({
        documentId: item.document_id,
        predictionId: item.prediction_id,
        finalLabel: selectedLabel,
        action,
      })
      setSubmittingIds((prev) => {
        const next = new Set(prev)
        next.delete(item.annotation_id)
        return next
      })
    } catch (error) {
      console.error('[Conflicts] submitAnnotation failed', error)
      setItems((prev) => [item, ...prev])
      setSubmittingIds((prev) => {
        const next = new Set(prev)
        next.delete(item.annotation_id)
        return next
      })
      setResolveError(
        `Could not resolve the conflict for "${item.predicted_label}" — item restored to the list.`,
      )
    }
  }

  return (
    <section className="mx-auto flex w-full max-w-4xl flex-col gap-6 py-4">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Reviewer Workspace
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Annotation Conflicts
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Two annotators disagreed on the same document — pick the final label.
        </p>
      </div>

      {resolveError ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
          {resolveError}
        </div>
      ) : null}

      {items === null ? (
        <div className="dashboard-card py-12 text-center text-sm text-slate-500">
          Loading conflicts...
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
      ) : items.length === 0 ? (
        <div className="dashboard-card border-dashed py-12 text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
            No conflicts
          </p>
          <p className="mt-3 text-sm text-slate-600">Nothing flagged for review right now.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((item) => (
            <ConflictListItem
              key={item.annotation_id}
              item={item}
              isSubmitting={submittingIds.has(item.annotation_id)}
              onResolve={handleResolve}
            />
          ))}
        </div>
      )}
    </section>
  )
}

export default ConflictsPage
