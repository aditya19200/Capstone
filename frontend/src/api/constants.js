// Mirrors backend/config/settings.py — Active Learning + review thresholds.
// The backend does not expose these over an endpoint yet, so this is a manual
// mirror. If Aditya changes settings.py, update these to match.

export const CONFIDENCE_HIGH = 0.85 // above this -> AUTO_ACCEPT
export const CONFIDENCE_LOW = 0.55 // below this -> ROUTE_TO_REVIEWER
export const REVIEW_THRESHOLD = 0.5 // confidence < this -> needs_review

// Mirrors LEGAL_LABELS in backend/models/response_models.py (id2label order).
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
