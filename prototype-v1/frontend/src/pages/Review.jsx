import { useEffect, useState } from 'react'
import { getLowConfidenceQueue, submitAnnotation } from '../api/client.js'
import ReviewQueueItem from '../components/review/ReviewQueueItem.jsx'
import ExplainPanel from '../components/explanation/ExplainPanel.jsx'

function ReviewPage() {
  const [items, setItems] = useState(null) // null = loading
  const [fetchError, setFetchError] = useState('')
  const [submitError, setSubmitError] = useState('')
  const [submittingIds, setSubmittingIds] = useState(() => new Set())
  const [explainingItem, setExplainingItem] = useState(null)

  // No setState before the first `await` here — calling this directly from
  // the effect below would otherwise trigger react-hooks/set-state-in-effect
  // (a synchronous setState inside an effect body forces an extra render).
  const fetchQueue = async () => {
    try {
      const queue = await getLowConfidenceQueue()
      setItems(queue)
      setFetchError('')
    } catch (error) {
      console.error('[Review] failed to load queue', error)
      setFetchError('Could not load the review queue. Please try again.')
      setItems([])
    }
  }

  useEffect(() => {
    ;(async () => {
      await fetchQueue()
    })()
  }, [])

  const ontologyCount = items?.filter((i) => i.flag_reason === 'ontology_conflict').length ?? 0
  const lowConfCount = (items?.length ?? 0) - ontologyCount

  const handleRetry = () => {
    setItems(null)
    setFetchError('')
    fetchQueue()
  }

  const handleSubmit = async (item, selectedLabel) => {
    // Guard against a double-click firing this twice for the same row before
    // React has re-rendered with the disabled button. Checking inside the
    // functional updater (not the outer `submittingIds` closure) makes this
    // race-proof: React applies queued updaters in call order against the
    // true latest state, even if both clicks fired before either committed.
    let alreadySubmitting = false
    setSubmittingIds((prev) => {
      if (prev.has(item.prediction_id)) {
        alreadySubmitting = true
        return prev
      }
      return new Set(prev).add(item.prediction_id)
    })

    if (alreadySubmitting) {
      return
    }

    setSubmitError('')
    setItems((prev) => prev.filter((row) => row.prediction_id !== item.prediction_id))

    const action = selectedLabel === item.predicted_label ? 'accept' : 'modify'

    try {
      await submitAnnotation({
        documentId: item.document_id,
        predictionId: item.prediction_id,
        finalLabel: selectedLabel,
        action,
      })
      setSubmittingIds((prev) => {
        const next = new Set(prev)
        next.delete(item.prediction_id)
        return next
      })
    } catch (error) {
      console.error('[Review] submitAnnotation failed', error)
      setItems((prev) => [item, ...prev])
      setSubmittingIds((prev) => {
        const next = new Set(prev)
        next.delete(item.prediction_id)
        return next
      })
      setSubmitError(
        `Could not submit the correction for "${item.predicted_label}" — item restored to the queue.`,
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
          Review Queue
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Confirm or correct predictions flagged for human review.
        </p>

        {/* Items arrive here for two unrelated reasons and the counts differ
            a lot between them, so showing the split is the difference between
            "some items" and knowing how much work is queued and why. */}
        {items?.length ? (
          <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
            <span className="rounded-full bg-slate-500/15 px-3 py-1 font-semibold text-slate-700 dark:text-slate-200">
              {items.length} to review
            </span>
            {ontologyCount > 0 ? (
              <span className="rounded-full bg-amber-500/15 px-3 py-1 font-medium text-amber-700 dark:text-amber-300">
                {ontologyCount} ontology conflict{ontologyCount === 1 ? '' : 's'}
              </span>
            ) : null}
            {lowConfCount > 0 ? (
              <span className="rounded-full bg-indigo-500/15 px-3 py-1 font-medium text-indigo-700 dark:text-indigo-300">
                {lowConfCount} low confidence
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      {submitError ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
          {submitError}
        </div>
      ) : null}

      {items === null ? (
        <div className="dashboard-card py-12 text-center text-sm text-slate-500">
          Loading review queue...
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
            All caught up
          </p>
          <p className="mt-3 text-sm text-slate-600">No items need review right now.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((item) => (
            <ReviewQueueItem
              key={item.prediction_id}
              item={item}
              isSubmitting={submittingIds.has(item.prediction_id)}
              onSubmit={handleSubmit}
              onExplain={setExplainingItem}
            />
          ))}
        </div>
      )}

      <ExplainPanel item={explainingItem} onClose={() => setExplainingItem(null)} />
    </section>
  )
}

export default ReviewPage
