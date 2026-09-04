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
import { BATCH_PAGE_SIZE } from './constants.js'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

// (A warnMockOnly() helper used to live here, flagging functions that had no
// backend route and always returned mock data. Every one of those now calls a
// real endpoint, so it had no callers left and was removed.)

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

export async function getRetrainStatus(jobId) {
  if (USE_MOCKS) return mocks.getRetrainStatus(jobId)
  const { data } = await axiosClient.get('/retrain/status', {
    params: jobId ? { job_id: jobId } : undefined,
  })
  return data
}

// How many validated annotations exist vs. how many retraining needs.
// Shown on the Retrain page so the threshold is visible before you click,
// and so newly submitted reviews visibly move the number.
export async function getRetrainEligibility() {
  if (USE_MOCKS) {
    return mocks.getRetrainEligibility
      ? mocks.getRetrainEligibility()
      : { validated_count: 0, required: 50, eligible: false }
  }
  const { data } = await axiosClient.get('/retrain/eligibility')
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
// /stats
// ---------------------------------------------------------------------------

// Everything the dashboard landing page needs, in one call. Deliberately not
// /admin/metrics — that route is admin-only, and the dashboard is also the
// landing page for annotators and reviewers.
export async function getDashboardStats() {
  if (USE_MOCKS) {
    return mocks.getDashboardStats ? mocks.getDashboardStats() : null
  }
  const { data } = await axiosClient.get('/stats/dashboard')
  return data
}

// ---------------------------------------------------------------------------
// /batches, /queue, /admin — real endpoints now exist (backend/routers/
// batches.py, queue.py, admin.py). Wired below, same USE_MOCKS pattern as
// everything above.
// ---------------------------------------------------------------------------

export async function submitBatchPaste(texts) {
  if (USE_MOCKS) return mocks.submitBatchPaste(texts)
  const { data } = await axiosClient.post('/batches/paste', { texts })
  return data
}

export async function submitBatchCsv(file) {
  if (USE_MOCKS) return mocks.submitBatchCsv(file)
  const form = new FormData()
  form.append('file', file)
  const { data } = await axiosClient.post('/batches/csv', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getBatch(batchId) {
  if (USE_MOCKS) return mocks.getBatch(batchId)
  const { data } = await axiosClient.get(`/batches/${batchId}`)
  return data
}

// Lists past batches, newest first — powers the "Recent batches" list so
// leaving the Annotate page (or logging out/back in) doesn't lose track of
// work. The underlying predictions were never actually lost — they live in
// the backend for as long as it keeps running — this just gives the UI a
// way to find them again instead of only knowing about the batch you just
// submitted this page-load.
export async function listBatches() {
  if (USE_MOCKS) return mocks.listBatches ? mocks.listBatches() : []
  const { data } = await axiosClient.get('/batches')
  return data.batches
}

export async function getBatchItems(batchId, page) {
  if (USE_MOCKS) return mocks.getBatchItems(batchId, page)
  const { data } = await axiosClient.get(`/batches/${batchId}/items`, {
    params: { page, page_size: BATCH_PAGE_SIZE },
  })
  return data
}

export async function exportBatchUrl(batchId) {
  if (USE_MOCKS) return mocks.exportBatchUrl(batchId)
  const base = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  return `${base}/batches/${batchId}/export`
}

export async function getLowConfidenceQueue() {
  if (USE_MOCKS) return mocks.getLowConfidenceQueue()
  // Real LowConfidenceItem calls the field text_excerpt; every consumer
  // (ReviewQueueItem.jsx) reads text_preview, matching the mock's naming —
  // without this rename the review queue silently shows no document text.
  const { data } = await axiosClient.get('/queue/low-confidence')
  return data.items.map((item) => ({
    prediction_id: item.prediction_id,
    text_preview: item.text_excerpt,
    predicted_label: item.predicted_label,
    confidence: item.confidence,
  }))
}

// Two field renames at the boundary rather than changes in the pages:
//   annotation_counts        -> annotation_counts_by_status  (Admin.jsx)
//   model_versions[].id      -> version_id                   (Admin.jsx keys
//                                                             rows on this and
//                                                             passes it to
//                                                             activate)
// version_number is passed through alongside so the table can show a readable
// label ("v-20260903-093923") instead of the raw uuid the activate endpoint
// needs. f1_per_class already matches and comes straight from the active
// model version's stored metrics.
export async function getAdminMetrics() {
  if (USE_MOCKS) return mocks.getAdminMetrics()
  const { data } = await axiosClient.get('/admin/metrics')
  return {
    ...data,
    annotation_counts_by_status: data.annotation_counts ?? {},
    model_versions: (data.model_versions ?? []).map((v) => ({
      version_id: v.id,
      version_number: v.version_number,
      accuracy: v.accuracy,
      is_active: v.is_active,
      trained_at: v.trained_at,
    })),
  }
}

// Speculative shape — GET /annotate has no has_conflict filter and its
// response item doesn't carry predicted_label or a text excerpt. See
// mocks.getConflicts for the exact gap.
export async function getConflicts() {
  if (USE_MOCKS) return mocks.getConflicts()
  // Real GET /annotate?has_conflict=true returns one row per annotation
  // (not paired by conflict, unlike the mock's second_annotation shape) —
  // ConflictListItem already renders correctly with second_annotation
  // absent, so no pairing logic needed here. text_excerpt -> text_preview
  // is the one field rename the component expects.
  const { data } = await axiosClient.get('/annotate', {
    params: { has_conflict: true },
  })
  return data.annotations.map((a) => ({
    annotation_id: a.annotation_id,
    prediction_id: a.prediction_id,
    document_id: null,
    text_preview: a.text_excerpt,
    predicted_label: a.predicted_label,
    final_label: a.final_label,
    user_id: a.user_id,
    annotation_status: a.annotation_status,
    annotated_at: a.annotated_at,
    has_conflict: a.has_conflict,
  }))
}

// versionId here is the model_versions uuid (mapped to version_id above),
// which is what the activate RPC matches on — not version_number.
export async function activateModelVersion(versionId) {
  if (USE_MOCKS) return mocks.activateModelVersion(versionId)
  const { data } = await axiosClient.post(`/admin/models/${versionId}/activate`)
  return data
}
