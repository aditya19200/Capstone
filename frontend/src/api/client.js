// API switchboard. One function per backend endpoint. Every consumer
// (pages/components) imports from here and never touches axiosClient or
// mocks.js directly, and never knows whether it's talking to the real
// backend or a mock.
//
// VITE_USE_MOCKS=true -> mocked responses (shaped like the real ones), with
// an artificial delay so loading states are visible.
// VITE_USE_MOCKS=false -> real calls to VITE_API_BASE_URL.
//
// Functions marked "mock-only" below have no backend route yet
// (backend/main.py only registers /predict, /explain, /annotate, /retrain,
// /ontology, /health) and always return mock data regardless of the flag.
// When Aditya adds the real route, give it a real branch here — nothing
// that calls these functions needs to change.

import axiosClient from './axiosClient.js'
import * as mocks from './mocks.js'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

const warnMockOnly = (fnName) => {
  if (import.meta.env.DEV) {
    console.warn(`[client] ${fnName}: no backend endpoint yet — always mocked`)
  }
}

// ---------------------------------------------------------------------------
// /predict
// ---------------------------------------------------------------------------

export async function predictText({ text, documentId } = {}) {
  if (USE_MOCKS) return mocks.predictText({ text, documentId })
  const { data } = await axiosClient.post('/predict', { text, document_id: documentId })
  return data
}

// Unused by any current screen (only relevant once a batch worker exists),
// included for completeness since the route already exists.
export async function getPredictionJob(jobId) {
  if (USE_MOCKS) return mocks.getPredictionJob(jobId)
  const { data } = await axiosClient.get(`/predict/${jobId}`)
  return data
}

// ---------------------------------------------------------------------------
// /explain
// ---------------------------------------------------------------------------

export async function requestExplain(predictionId) {
  if (USE_MOCKS) return mocks.requestExplain(predictionId)
  const { data } = await axiosClient.post('/explain', { prediction_id: predictionId })
  return data
}

// Polls by prediction_id (not the xai_job_id returned by requestExplain) —
// that's how GET /explain/{prediction_id} is keyed on the real backend.
export async function getExplain(predictionId) {
  if (USE_MOCKS) return mocks.getExplain(predictionId)
  const { data } = await axiosClient.get(`/explain/${predictionId}`)
  return data
}

// ---------------------------------------------------------------------------
// /annotate
// ---------------------------------------------------------------------------

export async function submitAnnotation({ documentId, predictionId, finalLabel, action, notes }) {
  if (USE_MOCKS) return mocks.submitAnnotation({ documentId, predictionId, finalLabel, action, notes })
  const { data } = await axiosClient.post('/annotate', {
    document_id: documentId,
    prediction_id: predictionId,
    final_label: finalLabel,
    action,
    notes,
  })
  return data
}

export async function listAnnotations({ documentId, status } = {}) {
  if (USE_MOCKS) return mocks.listAnnotations({ documentId, status })
  const { data } = await axiosClient.get('/annotate', {
    params: { document_id: documentId, status },
  })
  return data
}

// ---------------------------------------------------------------------------
// /retrain
// ---------------------------------------------------------------------------

export async function triggerRetrain({ notes, minAnnotations } = {}) {
  if (USE_MOCKS) return mocks.triggerRetrain({ notes, minAnnotations })
  const { data } = await axiosClient.post('/retrain', { notes, min_annotations: minAnnotations })
  return data
}

export async function getRetrainStatus() {
  if (USE_MOCKS) return mocks.getRetrainStatus()
  const { data } = await axiosClient.get('/retrain/status')
  return data
}

// ---------------------------------------------------------------------------
// /ontology
// ---------------------------------------------------------------------------

export async function getOntology() {
  if (USE_MOCKS) return mocks.getOntology()
  const { data } = await axiosClient.get('/ontology')
  return data
}

// Real substitute for CLAUDE.md's documented GET /concepts/{domain}.
export async function getOntologyNode(label) {
  if (USE_MOCKS) return mocks.getOntologyNode(label)
  const { data } = await axiosClient.get(`/ontology/${encodeURIComponent(label)}`)
  return data
}

export async function validateOntologyLabel({ label, parentLabel }) {
  if (USE_MOCKS) return mocks.validateOntologyLabel({ label, parentLabel })
  const { data } = await axiosClient.post('/ontology/validate', {
    label,
    parent_label: parentLabel,
  })
  return data
}

// ---------------------------------------------------------------------------
// /health
// ---------------------------------------------------------------------------

export async function getHealth() {
  if (USE_MOCKS) return mocks.getHealth()
  const { data } = await axiosClient.get('/health')
  return data
}

// ---------------------------------------------------------------------------
// Mock-only — no backend route exists yet. Shapes follow CLAUDE.md's
// originally documented contract (Task 2 batch flow, Task 4 queue, Task 5
// admin); treat as speculative until the real endpoints exist.
// ---------------------------------------------------------------------------

export async function submitBatchPaste(texts) {
  warnMockOnly('submitBatchPaste')
  return mocks.submitBatchPaste(texts)
}

export async function submitBatchCsv(file) {
  warnMockOnly('submitBatchCsv')
  return mocks.submitBatchCsv(file)
}

export async function getBatch(batchId) {
  warnMockOnly('getBatch')
  return mocks.getBatch(batchId)
}

export async function getBatchItems(batchId, page) {
  warnMockOnly('getBatchItems')
  return mocks.getBatchItems(batchId, page)
}

export async function exportBatchUrl(batchId) {
  warnMockOnly('exportBatchUrl')
  return mocks.exportBatchUrl(batchId)
}

export async function getLowConfidenceQueue() {
  warnMockOnly('getLowConfidenceQueue')
  return mocks.getLowConfidenceQueue()
}

export async function getAdminMetrics() {
  warnMockOnly('getAdminMetrics')
  return mocks.getAdminMetrics()
}

export async function activateModelVersion(versionId) {
  warnMockOnly('activateModelVersion')
  return mocks.activateModelVersion(versionId)
}
