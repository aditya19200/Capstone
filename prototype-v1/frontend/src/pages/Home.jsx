import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getDomains, getHealth } from '../api/client.js'
import { useAuth } from '../auth/AuthContext.jsx'

// The app used to open on a card reading "Login" with three buttons and no
// other text, so the first thing anyone saw gave no indication of what the
// system was. This is the public entry point: what it does, what is actually
// running right now, and then the same three role entries.
//
// /health and /domains are both reachable without a role, which is why the
// status strip can be on a page nobody has logged into yet.

const STEPS = [
  {
    n: '01',
    title: 'Ingest',
    body: 'CSV or PDF. Pages are split into passages that fit the model window.',
  },
  {
    n: '02',
    title: 'Classify',
    body: 'A domain-specific transformer labels each passage and reports its confidence.',
  },
  {
    n: '03',
    title: 'Check against the ontology',
    body: 'Concept markers mined from gold labels flag predictions whose vocabulary contradicts them — the confident mistakes confidence cannot catch.',
  },
  {
    n: '04',
    title: 'Route to a human',
    body: 'Low confidence or a contradiction sends the passage to the review queue. Everything else is accepted automatically.',
  },
  {
    n: '05',
    title: 'Retrain',
    body: 'Validated annotations become training data. The new version is registered, compared against the active one, and activated only on request.',
  },
]

const ROLES = [
  {
    role: 'annotator',
    title: 'Annotator',
    body: 'Upload documents and watch them classify.',
  },
  {
    role: 'reviewer',
    title: 'Reviewer',
    body: 'Work the queue of passages the system was unsure about, or flagged.',
  },
  {
    role: 'admin',
    title: 'Admin',
    body: 'Metrics, model versions, retraining and corpus export.',
  },
]

function StatusDot({ ok }) {
  return (
    <span
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${
        ok ? 'bg-green-500' : 'bg-slate-300'
      }`}
    />
  )
}

function StatusStrip() {
  const [health, setHealth] = useState(null)
  const [domains, setDomains] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [h, d] = await Promise.all([getHealth(), getDomains()])
        if (cancelled) return
        setHealth(h)
        setDomains(d)
      } catch {
        // Not fatal — the page is still worth showing with the backend down,
        // and saying so is more useful than an empty strip.
        if (!cancelled) setFailed(true)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  if (failed) {
    return (
      <p className="text-sm text-slate-500">
        The API is not responding. Start it with{' '}
        <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">
          ./run_local_demo.sh
        </code>
      </p>
    )
  }

  if (!health || !domains) {
    return <p className="text-sm text-slate-500">Checking what is running…</p>
  }

  const active = domains.domains?.find((d) => d.key === domains.active)
  const items = [
    {
      label: 'Model',
      value: health.model_version || 'baseline',
      ok: health.model_loaded,
    },
    {
      label: 'Domain',
      value: active ? `${active.display_name} · ${active.num_labels} labels` : domains.active,
      ok: Boolean(active),
    },
    {
      label: 'Database',
      value: health.data_layer?.supabase === 'live' ? 'Postgres live' : 'in-memory',
      ok: health.data_layer?.supabase === 'live',
    },
    {
      label: 'Graph',
      value: health.data_layer?.neo4j === 'live' ? 'Neo4j live' : 'in-memory',
      ok: health.data_layer?.neo4j === 'live',
    },
  ]

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-2">
          <StatusDot ok={item.ok} />
          <span className="text-xs uppercase tracking-wider text-slate-500">
            {item.label}
          </span>
          <span className="text-sm font-medium text-slate-700">{item.value}</span>
        </div>
      ))}
    </div>
  )
}

function HomePage() {
  const { mockLogin, isAuthenticated, role } = useAuth()
  const navigate = useNavigate()

  const enter = (role) => {
    mockLogin(role)
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto w-full max-w-5xl px-6 py-14">
        <header>
          <p className="section-kicker">Capstone Project</p>
          <h1 className="mt-4 max-w-3xl text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
            XAI-Enabled Knowledge Graph-Driven Annotation Framework
          </h1>
          <p className="mt-5 max-w-2xl text-lg text-slate-600">
            A human-in-the-loop system for labelling documents at scale. It
            classifies what it can, explains why, and routes what it should not
            decide alone to a person — then learns from those corrections.
          </p>
        </header>

        <section className="mt-10">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
            Open the workspace
          </h2>
          <p className="mt-2 text-sm text-slate-500">
            Each role sees a different part of the pipeline.
          </p>
          {isAuthenticated ? (
            // Coming back to "/" while signed in should not look like being
            // logged out. Picking another role still works below.
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              className="btn-primary mt-5"
            >
              Continue as {role} &rarr;
            </button>
          ) : null}
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            {ROLES.map((r) => (
              <button
                key={r.role}
                type="button"
                onClick={() => enter(r.role)}
                className="dashboard-card text-left transition hover:border-indigo-200"
              >
                <h3 className="text-base font-semibold text-slate-900">
                  {r.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">
                  {r.body}
                </p>
                <span className="mt-4 inline-block text-sm font-semibold text-indigo-600">
                  Enter →
                </span>
              </button>
            ))}
          </div>
        </section>

        <div className="dashboard-card mt-10">
          <p className="section-kicker mb-4">Running now</p>
          <StatusStrip />
        </div>

        <section className="mt-14">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
            How a document moves through it
          </h2>
          <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {STEPS.map((step) => (
              <li key={step.n} className="dashboard-card">
                <span className="text-sm font-semibold text-indigo-600">
                  {step.n}
                </span>
                <h3 className="mt-2 text-base font-semibold text-slate-900">
                  {step.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">
                  {step.body}
                </p>
              </li>
            ))}
          </ol>
        </section>

      </div>
    </div>
  )
}

export default HomePage
