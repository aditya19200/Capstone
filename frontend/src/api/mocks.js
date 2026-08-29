// Mock fixtures for client.js. Shapes for the first 12 functions mirror the
// real Pydantic response models in backend/models/response_models.py exactly.
// Shapes for the batch/queue/admin functions are speculative (no backend
// route exists yet) — update them once those endpoints are built.

import { LEGAL_LABELS, REVIEW_THRESHOLD } from './constants.js'

export const delay = (ms = 600) => new Promise((resolve) => setTimeout(resolve, ms))

let mockIdCounter = 0
const nextId = (prefix) => `${prefix}-mock-${++mockIdCounter}`

const routingDecisionFor = (confidence) => {
  if (confidence >= 0.85) return 'AUTO_ACCEPT'
  if (confidence >= 0.55) return 'NEEDS_EXPLANATION'
  return 'ROUTE_TO_REVIEWER'
}

const mockProbabilities = (topLabel, confidence) => {
  const rest = LEGAL_LABELS.filter((label) => label !== topLabel)
  const remaining = 1 - confidence
  const share = remaining / rest.length
  const probabilities = { [topLabel]: confidence }
  rest.forEach((label) => {
    probabilities[label] = Number(share.toFixed(4))
  })
  return probabilities
}

// ---------------------------------------------------------------------------
// /predict
// ---------------------------------------------------------------------------

export async function predictText({ text, documentId }) {
  await delay()
  const predictedLabel = LEGAL_LABELS[Math.floor(Math.random() * LEGAL_LABELS.length)]
  const confidence = Number((0.5 + Math.random() * 0.45).toFixed(4))
  const routingDecision = routingDecisionFor(confidence)

  return {
    prediction_id: nextId('pred'),
    document_id: documentId ?? null,
    predicted_label: predictedLabel,
    confidence,
    probabilities: mockProbabilities(predictedLabel, confidence),
    routing_decision: routingDecision,
    xai_job_id: routingDecision === 'AUTO_ACCEPT' ? null : nextId('xai'),
    entropy: Number((Math.random() * 1.5).toFixed(4)),
    margin: Number((Math.random() * 0.5).toFixed(4)),
    text,
  }
}

export async function getPredictionJob(jobId) {
  await delay()
  return {
    job_id: jobId,
    status: 'completed',
    result: await predictText({ text: '(mock queued prediction)' }),
  }
}

// ---------------------------------------------------------------------------
// /explain
// ---------------------------------------------------------------------------

export async function requestExplain(predictionId) {
  await delay(300)
  return {
    xai_job_id: nextId('xai'),
    prediction_id: predictionId,
    status: 'pending',
    message: `SHAP explanation job queued for prediction '${predictionId}'. Poll GET /explain/{prediction_id} for the result.`,
  }
}

const explainStatusByPrediction = new Map()

export async function getExplain(predictionId) {
  await delay(800)

  const callCount = (explainStatusByPrediction.get(predictionId) || 0) + 1
  explainStatusByPrediction.set(predictionId, callCount)

  if (callCount < 2) {
    return {
      explanation_id: nextId('exp'),
      prediction_id: predictionId,
      status: 'pending',
      token_importances: null,
      generated_at: null,
    }
  }

  const sampleTokens = ['The', 'agreement', 'between', 'the', 'parties', 'creates', 'binding', 'obligations']
  return {
    explanation_id: nextId('exp'),
    prediction_id: predictionId,
    status: 'done',
    token_importances: sampleTokens.map((token) => ({
      token,
      importance: Number((Math.random() * 2 - 1).toFixed(3)),
    })),
    generated_at: new Date().toISOString(),
  }
}

// ---------------------------------------------------------------------------
// /annotate
// ---------------------------------------------------------------------------

const mockAnnotations = []

export async function submitAnnotation({ documentId, predictionId, finalLabel, action, notes }) {
  await delay(400)
  const conflictDetected = action === 'modify' && Math.random() > 0.7

  const record = {
    annotation_id: nextId('ann'),
    document_id: documentId,
    prediction_id: predictionId,
    final_label: finalLabel,
    action,
    annotation_status: action === 'reject' ? 'rejected' : 'validated',
    conflict_detected: conflictDetected,
    message:
      action === 'reject'
        ? 'Document rejected and marked out of scope.'
        : conflictDetected
          ? `Annotation recorded with label '${finalLabel}', but a conflict was detected. The annotation has been flagged for Reviewer resolution.`
          : `Annotation recorded with label '${finalLabel}'.`,
  }

  mockAnnotations.unshift({
    id: record.annotation_id,
    document_id: documentId,
    annotator_id: 'mock-user-1',
    validated_label: finalLabel,
    status: record.annotation_status,
    annotated_at: new Date().toISOString(),
    notes: notes ?? null,
  })

  return record
}

export async function listAnnotations({ documentId, status } = {}) {
  await delay()
  const filtered = mockAnnotations.filter((row) => {
    if (documentId && row.document_id !== documentId) return false
    if (status && row.status !== status) return false
    return true
  })

  return {
    total: filtered.length,
    annotations: filtered.map((row) => ({
      annotation_id: row.id,
      document_id: row.document_id,
      user_id: row.annotator_id,
      final_label: row.validated_label,
      annotation_status: row.status,
      annotated_at: row.annotated_at,
    })),
  }
}

// ---------------------------------------------------------------------------
// /retrain
// ---------------------------------------------------------------------------

let mockRetrainJob = null

export async function triggerRetrain({ notes, minAnnotations = 50 } = {}) {
  await delay(400)
  mockRetrainJob = {
    retrain_job_id: nextId('retrain'),
    status: 'running',
    model_version: null,
    accuracy: null,
    started_at: new Date().toISOString(),
    completed_at: null,
    message: `Retraining job queued (min_annotations=${minAnnotations}${notes ? `, notes="${notes}"` : ''}).`,
  }
  return {
    retrain_job_id: mockRetrainJob.retrain_job_id,
    status: mockRetrainJob.status,
    message: mockRetrainJob.message,
  }
}

export async function getRetrainStatus() {
  await delay()
  if (!mockRetrainJob) {
    return {
      retrain_job_id: null,
      status: 'complete',
      model_version: 'v1',
      accuracy: 0.874,
      started_at: null,
      completed_at: null,
      message: 'No retraining job has been triggered yet.',
    }
  }
  return mockRetrainJob
}

// ---------------------------------------------------------------------------
// /ontology
// ---------------------------------------------------------------------------

const mockOntologyNodes = [
  { label: 'Contract Law', description: 'Agreements and enforceable obligations.', parent: null, children: ['Breach', 'Consideration'], related: ['Corporate / Company Law'] },
  { label: 'Criminal Law', description: 'Offenses against the state.', parent: null, children: ['Mens Rea', 'Evidence'], related: ['Civil Procedure / Other'] },
  { label: 'Breach', description: 'Failure to perform a contractual obligation.', parent: 'Contract Law', children: [], related: ['Consideration'] },
  { label: 'Consideration', description: 'Value exchanged between contracting parties.', parent: 'Contract Law', children: [], related: ['Breach'] },
]

export async function getOntology() {
  await delay()
  return { total: mockOntologyNodes.length, nodes: mockOntologyNodes }
}

export async function getOntologyNode(label) {
  await delay()
  return (
    mockOntologyNodes.find((node) => node.label === label) || {
      label,
      description: null,
      parent: null,
      children: [],
      related: [],
    }
  )
}

export async function validateOntologyLabel({ label, parentLabel }) {
  await delay(300)
  const node = mockOntologyNodes.find((item) => item.label === label)
  return {
    label,
    valid: Boolean(node) || LEGAL_LABELS.includes(label),
    parent_label: parentLabel ?? null,
    parent_valid: parentLabel ? mockOntologyNodes.some((item) => item.label === parentLabel) : null,
    message: node || LEGAL_LABELS.includes(label) ? 'Label exists in the ontology.' : 'Label not found.',
  }
}

// ---------------------------------------------------------------------------
// /health
// ---------------------------------------------------------------------------

export async function getHealth() {
  await delay(100)
  return { status: 'ok', model_loaded: true, model_path: '(mock)' }
}

// ---------------------------------------------------------------------------
// Mock-only — no backend route exists yet (Task 2 batch flow, Task 4 queue,
// Task 5 admin). Shapes follow CLAUDE.md's originally documented contract;
// treat as speculative until Aditya builds the real endpoints.
// ---------------------------------------------------------------------------

const mockBatches = new Map()

export async function submitBatchPaste(texts) {
  await delay(300)
  const batchId = nextId('batch')
  mockBatches.set(batchId, { total: texts.length, completed: 0, items: texts })
  return { batch_id: batchId, total_items: texts.length }
}

export async function submitBatchCsv(file) {
  await delay(300)
  const batchId = nextId('batch')
  mockBatches.set(batchId, { total: 10, completed: 0, items: [], fileName: file?.name })
  return { batch_id: batchId, total_items: 10 }
}

export async function getBatch(batchId) {
  await delay(500)
  const batch = mockBatches.get(batchId)
  const total = batch?.total ?? 10
  const completed = Math.min(total, (batch?.completed ?? 0) + 3)
  if (batch) batch.completed = completed
  return {
    status: completed >= total ? 'done' : 'processing',
    total_items: total,
    completed_items: completed,
  }
}

export async function getBatchItems(batchId, page = 1) {
  await delay()
  const pageSize = 10
  const items = Array.from({ length: pageSize }, (_, index) => {
    const label = LEGAL_LABELS[index % LEGAL_LABELS.length]
    const confidence = Number((0.4 + Math.random() * 0.55).toFixed(4))
    return {
      id: `${batchId}-item-${(page - 1) * pageSize + index + 1}`,
      seq: (page - 1) * pageSize + index + 1,
      text_content: '(mock legal text excerpt)',
      predicted_label: label,
      confidence,
      status: confidence < REVIEW_THRESHOLD ? 'pending' : 'classified',
    }
  })
  return { batch_id: batchId, page, page_size: pageSize, items }
}

export async function exportBatchUrl(batchId) {
  await delay(100)
  return `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/batches/${batchId}/export`
}

export async function getLowConfidenceQueue() {
  await delay()
  return Array.from({ length: 6 }, (_, index) => {
    const label = LEGAL_LABELS[index % LEGAL_LABELS.length]
    return {
      prediction_id: nextId('pred'),
      text_preview: '(mock excerpt requiring human review)',
      predicted_label: label,
      confidence: Number((0.2 + Math.random() * 0.3).toFixed(4)),
    }
  })
}

export async function getAdminMetrics() {
  await delay()
  return {
    f1_per_class: Object.fromEntries(LEGAL_LABELS.map((label) => [label, Number((0.6 + Math.random() * 0.35).toFixed(3))])),
    annotation_counts_by_status: { pending: 42, validated: 318, rejected: 12 },
    model_versions: [
      { version_id: 'v1', accuracy: 0.874, trained_at: '2026-06-01T00:00:00Z', is_active: true },
      { version_id: 'v2-candidate', accuracy: 0.891, trained_at: '2026-08-20T00:00:00Z', is_active: false },
    ],
  }
}

export async function activateModelVersion(versionId) {
  await delay(300)
  return { success: true, version_id: versionId, message: `Model version '${versionId}' activated (mock).` }
}
