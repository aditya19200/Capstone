import { useEffect, useRef, useState } from 'react'
import { getRetrainStatus, triggerRetrain } from '../api/client.js'

const POLL_INTERVAL_MS = 3000
const MAX_POLL_MS = 5 * 60 * 1000

function Spinner() {
  return (
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600" />
  )
}

// FastAPI's HTTPException detail is already a clear, specific sentence
// ("Insufficient validated annotations for retraining. Found 32, required
// 50.") — surface it directly instead of re-deriving our own message.
const extractErrorDetail = (error, fallback) => error?.response?.data?.detail || fallback

function RetrainPage() {
  // phase: 'checking' | 'error' | 'idle' | 'polling' | 'timeout' | 'done' | 'job-failed'
  const [phase, setPhase] = useState('checking')
  const [checkError, setCheckError] = useState('')
  const [jobId, setJobId] = useState(null)
  const [lastResult, setLastResult] = useState(null)
  const [notes, setNotes] = useState('')
  const [isTriggering, setIsTriggering] = useState(false)
  const [triggerError, setTriggerError] = useState('')

  const pollStartRef = useRef(null)

  // On mount: check whether a job is already pending/running (started by
  // this admin in an earlier session, or by a teammate) before showing the
  // trigger button over it.
  useEffect(() => {
    ;(async () => {
      try {
        const result = await getRetrainStatus()
        if (result.retrain_job_id && (result.status === 'pending' || result.status === 'running')) {
          setJobId(result.retrain_job_id)
          pollStartRef.current = Date.now()
          setPhase('polling')
        } else {
          if (result.retrain_job_id) {
            setLastResult(result)
          }
          setPhase('idle')
        }
      } catch (error) {
        if (error?.response?.status === 404) {
          // No retraining jobs have ever been triggered — normal, not an error.
          setPhase('idle')
          return
        }
        console.error('[Retrain] initial status check failed', error)
        setCheckError(extractErrorDetail(error, 'Could not check retraining status.'))
        setPhase('error')
      }
    })()
  }, [])

  useEffect(() => {
    if (phase !== 'polling' || !jobId) {
      return undefined
    }

    const interval = setInterval(async () => {
      if (Date.now() - pollStartRef.current > MAX_POLL_MS) {
        clearInterval(interval)
        setPhase('timeout')
        return
      }

      try {
        const result = await getRetrainStatus(jobId)
        if (result.status === 'complete') {
          clearInterval(interval)
          setLastResult(result)
          setPhase('done')
        } else if (result.status === 'failed') {
          clearInterval(interval)
          setLastResult(result)
          setPhase('job-failed')
        }
        // pending / running -> keep polling
      } catch (error) {
        console.error('[Retrain] status polling failed', error)
        clearInterval(interval)
        setCheckError(extractErrorDetail(error, 'Lost connection while checking retraining status.'))
        setPhase('error')
      }
    }, POLL_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [phase, jobId])

  const handleTrigger = async () => {
    setIsTriggering(true)
    setTriggerError('')

    try {
      const result = await triggerRetrain({ notes: notes.trim() || undefined })
      setJobId(result.retrain_job_id)
      setLastResult(null)
      pollStartRef.current = Date.now()
      setPhase('polling')
    } catch (error) {
      console.error('[Retrain] trigger failed', error)
      setTriggerError(extractErrorDetail(error, 'Could not trigger retraining. Please try again.'))
    } finally {
      setIsTriggering(false)
    }
  }

  const handleCheckAgain = () => {
    pollStartRef.current = Date.now()
    setPhase('polling')
  }

  const handleRunAnother = () => {
    setJobId(null)
    setLastResult(null)
    setNotes('')
    setPhase('idle')
  }

  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-4">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
          Admin Workspace
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
          Trigger Retraining
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
          Kick off a retraining run using currently validated annotations.
        </p>
      </div>

      {phase === 'checking' ? (
        <div className="dashboard-card flex flex-col items-center gap-4 py-12 text-center">
          <Spinner />
          <p className="text-sm font-medium text-slate-700">Checking retraining status...</p>
        </div>
      ) : null}

      {phase === 'error' ? (
        <div className="dashboard-card border-dashed text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-red-600">
            Something went wrong
          </p>
          <p className="mt-3 text-sm text-slate-600">{checkError}</p>
        </div>
      ) : null}

      {phase === 'idle' ? (
        <div className="dashboard-card space-y-5">
          {lastResult ? (
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
              Last run: {lastResult.message}
            </div>
          ) : null}

          <div>
            <label
              htmlFor="retrain-notes"
              className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500"
            >
              Notes (optional)
            </label>
            <textarea
              id="retrain-notes"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              disabled={isTriggering}
              rows={3}
              placeholder="Context for this retraining run..."
              className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100"
            />
          </div>

          {triggerError ? (
            <p className="text-sm font-medium text-red-600">{triggerError}</p>
          ) : null}

          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleTrigger}
              disabled={isTriggering}
              className="btn-primary px-6 py-3"
            >
              {isTriggering ? 'Triggering...' : 'Trigger Retrain'}
            </button>
          </div>
        </div>
      ) : null}

      {phase === 'polling' ? (
        <div className="dashboard-card flex flex-col items-center gap-4 py-12 text-center">
          <Spinner />
          <p className="text-sm font-medium text-slate-700">Retraining in progress...</p>
          <p className="text-xs uppercase tracking-[0.16em] text-slate-400">Job {jobId}</p>
        </div>
      ) : null}

      {phase === 'timeout' ? (
        <div className="dashboard-card border-dashed text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">
            Taking longer than expected
          </p>
          <p className="mt-3 text-sm text-slate-600">
            This retraining run has been going for a while. It may still complete — you can check
            again.
          </p>
          <button type="button" onClick={handleCheckAgain} className="btn-primary mt-4 px-6 py-3">
            Check Again
          </button>
        </div>
      ) : null}

      {phase === 'done' ? (
        <div className="dashboard-card border-dashed text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-green-600">
            Retraining complete
          </p>
          <p className="mt-3 text-sm text-slate-600">{lastResult?.message}</p>
          <button type="button" onClick={handleRunAnother} className="btn-secondary mt-4 px-6 py-3">
            Trigger Another Run
          </button>
        </div>
      ) : null}

      {phase === 'job-failed' ? (
        <div className="dashboard-card border-dashed text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-red-600">
            Retraining failed
          </p>
          <p className="mt-3 text-sm text-slate-600">{lastResult?.message}</p>
          <button type="button" onClick={handleRunAnother} className="btn-primary mt-4 px-6 py-3">
            Try Again
          </button>
        </div>
      ) : null}
    </section>
  )
}

export default RetrainPage
