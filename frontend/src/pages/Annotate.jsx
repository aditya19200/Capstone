import { useState } from 'react'
import PredictionCard from '../components/prediction/PredictionCard.jsx'

const mockPredictionResult = {
  predictedLabel: 'Contract Law',
  confidenceScore: 0.87,
  routingDecision: 'AUTO_ACCEPT',
}

const isCsvFile = (file) => {
  if (!file) {
    return false
  }

  const lowerCaseName = file.name.toLowerCase()
  return lowerCaseName.endsWith('.csv') || file.type === 'text/csv'
}

function AnnotatePage() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [predictionResult, setPredictionResult] = useState(null)

  const handleFileSelection = (file) => {
    if (!file) {
      return
    }

    if (!isCsvFile(file)) {
      setSelectedFile(null)
      setPredictionResult(null)
      setErrorMessage('Please upload a valid CSV file.')
      return
    }

    setSelectedFile(file)
    setPredictionResult(null)
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

  const handlePredict = () => {
    setPredictionResult(mockPredictionResult)
  }

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
          Upload dataset to preview the batch prediction workflow and routing decision.
        </p>
      </div>

      <div className="dashboard-card">
        <div className="border-b border-slate-200 pb-5">
          <p className="section-kicker">Dataset Upload</p>
          <h3 className="mt-2 text-xl font-semibold text-slate-900">
            Upload Legal Dataset
          </h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            Select a CSV file containing legal records to simulate batch annotation predictions.
          </p>
        </div>

        <div className="mt-6 space-y-5">
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
            <p className="mt-2 text-sm text-slate-600">
              or click to browse from your device
            </p>
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
            {errorMessage ? (
              <p className="mt-2 text-sm font-medium text-red-600">{errorMessage}</p>
            ) : null}
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={handlePredict}
              disabled={!selectedFile}
              className="btn-primary px-6 py-3"
            >
              Upload & Predict
            </button>
          </div>
        </div>
      </div>

      {predictionResult ? (
        <PredictionCard
          predictedLabel={predictionResult.predictedLabel}
          confidenceScore={predictionResult.confidenceScore}
          routingDecision={predictionResult.routingDecision}
        />
      ) : (
        <div className="dashboard-card border-dashed p-8 text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
            Prediction Result
          </p>
          <p className="mt-3 text-sm text-slate-600">
            Upload a CSV file and click Upload & Predict to view the mock classification output.
          </p>
        </div>
      )}
    </section>
  )
}

export default AnnotatePage
