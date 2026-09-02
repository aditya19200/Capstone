import { useEffect, useRef, useState } from 'react'
import { exportBatchUrl, getBatch, getBatchItems, submitBatchCsv, submitBatchPaste } from '../api/client.js'
import BatchItemsTable from '../components/annotation/BatchItemsTable.jsx'
import BatchProgress from '../components/annotation/BatchProgress.jsx'
import ConceptsPanel from '../components/annotation/ConceptsPanel.jsx'
import LegalTextInput from '../components/annotation/LegalTextInput.jsx'
import ExplainPanel from '../components/explanation/ExplainPanel.jsx'

const POLL_INTERVAL_MS = 3000
const MAX_POLL_MS = 5 * 60 * 1000

const isCsvFile = (file) => {
  if (!file) {
    return false
  }

  const lowerCaseName = file.name.toLowerCase()
  return lowerCaseName.endsWith('.csv') || file.type === 'text/csv'
}

function AnnotatePage() {
  const [mode, setMode] = useState('paste')
  const [pasteText, setPasteText] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  // phase: 'idle' | 'processing' | 'done' | 'error' | 'timeout'
  const [phase, setPhase] = useState('idle')
  const [batchId, setBatchId] = useState(null)
  const [totalItems, setTotalItems] = useState(0)
  const [completedItems, setCompletedItems] = useState(0)
  const [items, setItems] = useState([])
  const [page, setPage] = useState(1)
  const [isLoadingItems, setIsLoadingItems] = useState(false)
  const [explainingItem, setExplainingItem] = useState(null)
  const [conceptsItem, setConceptsItem] = useState(null)
  const [exportUrl, setExportUrl] = useState(null)

  const pollStartRef = useRef(null)

  const handleFileSelection = (file) => {
    if (!file) {
      return
    }

    if (!isCsvFile(file)) {
      setSelectedFile(null)
      setErrorMessage('Please upload a valid CSV file.')
      return
    }

    setSelectedFile(file)
    setErrorMessage('')
  }

  const handleFileChange = (event) => {
    handleFileSelection(event.target.files?.[0] || null)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    handleFileSelection(event.dataTransfer.files?.[0] || null)
  }

  const resetToIdle = () => {
    setPhase('idle')
    setBatchId(null)
    setTotalItems(0)
    setCompletedItems(0)
    setItems([])
    setPage(1)
    setPasteText('')
    setSelectedFile(null)
    setErrorMessage('')
    setExportUrl(null)
  }

  const handleSubmit = async () => {
    setErrorMessage('')

    try {
      let response

      if (mode === 'paste') {
        const texts = pasteText
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean)

        if (texts.length === 0) {
          setErrorMessage('Paste at least one line of legal text (one document per line).')
          return
        }

        response = await submitBatchPaste(texts)
      } else {
        if (!selectedFile) {
          setErrorMessage('Select a CSV file first.')
          return
        }

        response = await submitBatchCsv(selectedFile)
      }

      setBatchId(response.batch_id)
      setTotalItems(response.total_items)
      setCompletedItems(0)
      pollStartRef.current = Date.now()
      setPhase('processing')
    } catch (error) {
      console.error('[Annotate] batch submit failed', error)
      setErrorMessage('Something went wrong submitting the batch. Please try again.')
      setPhase('error')
    }
  }

  useEffect(() => {
    if (phase !== 'processing' || !batchId) {
      return undefined
    }

    const interval = setInterval(async () => {
      if (Date.now() - pollStartRef.current > MAX_POLL_MS) {
        clearInterval(interval)
        setPhase('timeout')
        return
      }

      try {
        const batch = await getBatch(batchId)
        setCompletedItems(batch.completed_items)
        setTotalItems(batch.total_items)

        if (batch.status === 'done') {
          clearInterval(interval)
          const itemsResponse = await getBatchItems(batchId, 1)
          setItems(itemsResponse.items)
          setPage(1)
          setPhase('done')
        }
      } catch (error) {
        console.error('[Annotate] batch polling failed', error)
        clearInterval(interval)
        setErrorMessage('Lost connection while checking batch progress.')
        setPhase('error')
      }
    }, POLL_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [phase, batchId])

  useEffect(() => {
    if (phase !== 'done' || !batchId) {
      return
    }

    // exportBatchUrl is mock-only until the backend /batches/{id}/export route exists.
    if (import.meta.env.VITE_USE_MOCKS !== 'true') {
      setExportUrl(null)
      return
    }

    ;(async () => {
      try {
        const url = await exportBatchUrl(batchId)
        setExportUrl(url)
      } catch (error) {
        console.error('[Annotate] exportBatchUrl failed', error)
      }
    })()
  }, [phase, batchId])

  const handlePageChange = async (nextPage) => {
    setIsLoadingItems(true)
    try {
      const itemsResponse = await getBatchItems(batchId, nextPage)
      setItems(itemsResponse.items)
      setPage(nextPage)
    } finally {
      setIsLoadingItems(false)
    }
  }

  const handleCheckAgain = () => {
    pollStartRef.current = Date.now()
    setPhase('processing')
  }

  const handleExplain = (item) => {
    setExplainingItem(item)
  }

  const handleConcepts = (item) => {
    setConceptsItem(item)
  }

  const isSubmitting = phase === 'processing'

  return (
    <section className="mx-auto flex w-full max-w-4xl flex-col gap-8 py-4">
      <div className="text-center">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Annotator Workspace
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Annotate Legal Text
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Paste text or upload a CSV to classify a batch of legal documents.
        </p>
      </div>

      {phase === 'idle' || phase === 'error' ? (
        <div className="dashboard-card">
          <div className="flex gap-3 border-b border-slate-200 pb-5">
            <button
              type="button"
              onClick={() => setMode('paste')}
              className={mode === 'paste' ? 'btn-primary' : 'btn-secondary'}
            >
              Paste Text
            </button>
            <button
              type="button"
              onClick={() => setMode('csv')}
              className={mode === 'csv' ? 'btn-primary' : 'btn-secondary'}
            >
              Upload CSV
            </button>
          </div>

          <div className="mt-6 space-y-5">
            {mode === 'paste' ? (
              <LegalTextInput
                value={pasteText}
                onChange={(event) => setPasteText(event.target.value)}
                placeholder="Paste one legal document per line..."
              />
            ) : (
              <>
                <label
                  htmlFor="legal-dataset-upload"
                  onDragOver={(event) => {
                    event.preventDefault()
                    setIsDragging(true)
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className={[
                    'flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition',
                    isDragging
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-slate-100',
                  ].join(' ')}
                >
                  <div className="rounded-full bg-white p-4 shadow-sm">
                    <div className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-indigo-700">
                      CSV
                    </div>
                  </div>
                  <h4 className="mt-5 text-lg font-semibold text-slate-900">
                    Drag and drop your CSV file here
                  </h4>
                  <p className="mt-2 text-sm text-slate-600">or click to browse from your device</p>
                  <p className="mt-4 text-xs uppercase tracking-[0.2em] text-slate-400">
                    Supported format: .csv
                  </p>
                </label>

                <input
                  id="legal-dataset-upload"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleFileChange}
                  className="sr-only"
                />

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                    Selected File
                  </p>
                  <p className="mt-2 text-sm font-medium text-slate-900">
                    {selectedFile ? selectedFile.name : 'No file selected'}
                  </p>
                </div>
              </>
            )}

            {errorMessage ? (
              <p className="text-sm font-medium text-red-600">{errorMessage}</p>
            ) : null}

            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="btn-primary px-6 py-3"
              >
                Submit Batch
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {phase === 'processing' ? (
        <BatchProgress completedItems={completedItems} totalItems={totalItems} />
      ) : null}

      {phase === 'timeout' ? (
        <div className="dashboard-card border-dashed text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
            Taking longer than expected
          </p>
          <p className="mt-3 text-sm text-slate-600">
            This batch has been processing for a while. It may still complete — you can check
            again or come back later.
          </p>
          <button type="button" onClick={handleCheckAgain} className="btn-primary mt-4 px-6 py-3">
            Check Again
          </button>
        </div>
      ) : null}

      {phase === 'done' ? (
        <>
          <BatchItemsTable
            items={items}
            page={page}
            totalItems={totalItems}
            isLoading={isLoadingItems}
            onPageChange={handlePageChange}
            onExplain={handleExplain}
            onConcepts={handleConcepts}
            exportUrl={exportUrl}
          />
          <div className="flex justify-end">
            <button type="button" onClick={resetToIdle} className="btn-secondary">
              Start New Batch
            </button>
          </div>
        </>
      ) : null}

      <ExplainPanel item={explainingItem} onClose={() => setExplainingItem(null)} />
      <ConceptsPanel item={conceptsItem} onClose={() => setConceptsItem(null)} />
    </section>
  )
}

export default AnnotatePage
