import { useEffect, useState } from 'react'
import { getOntologyNode } from '../../api/client.js'

// Simple in-memory cache keyed by label — only 10 labels ever exist, so this
// avoids re-fetching the same node every time a row's Concepts button is
// reopened within the session.
const nodeCache = new Map()

function Spinner() {
  return (
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600" />
  )
}

function ConceptList({ title, labels }) {
  if (!labels || labels.length === 0) {
    return null
  }

  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{title}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {labels.map((label) => (
          <span
            key={label}
            className="rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
          >
            {label}
          </span>
        ))}
      </div>
    </div>
  )
}

function ConceptsPanel({ item, onClose }) {
  const label = item?.predicted_label
  const [status, setStatus] = useState('idle') // 'idle' | 'loading' | 'done' | 'failed'
  const [node, setNode] = useState(null)
  const [error, setError] = useState(null)

  const fetchNode = async (targetLabel) => {
    if (nodeCache.has(targetLabel)) {
      setNode(nodeCache.get(targetLabel))
      setStatus('done')
      return
    }

    setStatus('loading')
    setError(null)

    try {
      const result = await getOntologyNode(targetLabel)
      nodeCache.set(targetLabel, result)
      setNode(result)
      setStatus('done')
    } catch (fetchError) {
      console.error('[ConceptsPanel] getOntologyNode failed', fetchError)
      setError('Could not load related concepts for this label.')
      setStatus('failed')
    }
  }

  useEffect(() => {
    if (label) {
      ;(async () => {
        await fetchNode(label)
      })()
    }
  }, [label])

  if (!item) {
    return null
  }

  const isEmpty =
    status === 'done' &&
    node &&
    !node.description &&
    !node.parent &&
    (!node.children || node.children.length === 0) &&
    (!node.related || node.related.length === 0)

  return (
    <>
      <button
        type="button"
        aria-label="Close concepts panel backdrop"
        className="fixed inset-0 z-30 bg-slate-950/30 backdrop-blur-sm"
        onClick={onClose}
      />

      <aside className="fixed bottom-0 right-0 top-20 z-40 w-full max-w-md overflow-y-auto border-l border-slate-200 bg-slate-50 shadow-2xl">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/95 px-6 py-5 backdrop-blur">
          <div>
            <p className="section-kicker">Related Concepts</p>
            <h3 className="mt-2 text-xl font-semibold text-slate-900">{label}</h3>
          </div>
          <button type="button" onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>

        <div className="space-y-6 p-6">
          {status === 'loading' && (
            <div className="dashboard-card flex flex-col items-center gap-4 py-12 text-center">
              <Spinner />
              <p className="text-sm font-medium text-slate-700">Loading related concepts...</p>
            </div>
          )}

          {status === 'failed' && (
            <div className="dashboard-card border-dashed text-center">
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-red-600">
                Something went wrong
              </p>
              <p className="mt-3 text-sm text-slate-600">{error}</p>
              <button
                type="button"
                onClick={() => fetchNode(label)}
                className="btn-primary mt-4 px-6 py-3"
              >
                Retry
              </button>
            </div>
          )}

          {status === 'done' && isEmpty && (
            <div className="dashboard-card border-dashed text-center">
              <p className="text-sm text-slate-600">
                No related concepts found for this label.
              </p>
            </div>
          )}

          {status === 'done' && node && !isEmpty && (
            <div className="dashboard-card space-y-6">
              {node.description ? (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                    Description
                  </p>
                  <p className="mt-2 text-sm text-slate-700">{node.description}</p>
                </div>
              ) : null}

              <ConceptList title="Parent concept" labels={node.parent ? [node.parent] : []} />
              <ConceptList title="Child concepts" labels={node.children} />
              <ConceptList title="Related concepts" labels={node.related} />
            </div>
          )}
        </div>
      </aside>
    </>
  )
}

export default ConceptsPanel
