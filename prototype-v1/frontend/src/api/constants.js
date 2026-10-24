// Mirrors backend/config/settings.py — Active Learning + review thresholds.
// The backend does not expose these over an endpoint yet, so this is a manual
// mirror. If Aditya changes settings.py, update these to match.

export const CONFIDENCE_HIGH = 0.85 // above this -> AUTO_ACCEPT
export const CONFIDENCE_LOW = 0.55 // below this -> ROUTE_TO_REVIEWER
export const REVIEW_THRESHOLD = 0.55 // confidence < this -> Review queue; equal to CONFIDENCE_LOW

// Batch results table page size. Must match what client.js actually sends
// as page_size on GET /batches/{id}/items — the two drifting apart is what
// made pagination lie about how many pages existed.
export const BATCH_PAGE_SIZE = 15

// FALLBACK ONLY. The label dropdowns read GET /domains via hooks/useLabels.js,
// which is fed by the backend's domain pack — this list is what they fall back
// to when that call fails, so a reviewer mid-queue still gets a usable select.
// mocks.js also uses it as its fixture label set.
//
// It does not need to track the backend label-for-label any more; it only
// needs to be a sane default for the legal deployment.
export const LEGAL_LABELS = [
  'Contract Law',
  'Criminal Law',
  'Constitutional Law',
  'Corporate / Company Law',
  'Property / Real Estate Law',
  'Family Law',
  'Labour & Employment Law',
  'Intellectual Property Law',
  'Taxation Law',
  'Civil Procedure / Other',
]
