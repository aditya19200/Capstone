import { useLocation } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext.jsx'
import useDomain from '../../hooks/useDomain.js'

// Each domain is served by its own instance: the label set is fixed when a
// backend process starts, so one process cannot serve both. This makes the two
// instances read as one product — pick a domain, land on the same page in it,
// still signed in as the same role.
//
// Where the instances live comes from VITE_DOMAIN_LINKS, set by
// run_local_demo.sh ("legal=http://localhost:5180,medical=http://localhost:5181").
// Unset — any single-domain deployment — and the switcher does not render.
function parseLinks(raw) {
  return (raw || '')
    .split(',')
    .map((pair) => pair.split('='))
    .filter(([key, url]) => key && url)
    .map(([key, url]) => ({ key: key.trim(), url: url.trim().replace(/\/$/, '') }))
}

const LINKS = parseLinks(import.meta.env.VITE_DOMAIN_LINKS)

// The other domain is another page load, so fade this one out first (the
// .page-leaving rule in index.css) and let the next one fade in. A modified
// click — new tab, new window — is left to the browser.
const LEAVE_MS = 160

function leaveTo(event, target) {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
  event.preventDefault()
  document.documentElement.classList.add('page-leaving')
  window.setTimeout(() => {
    window.location.href = target
  }, LEAVE_MS)
}

// Coming back with the Back button can restore this page exactly as it was
// left — faded out — so undo it whenever the page is shown.
window.addEventListener('pageshow', () => {
  document.documentElement.classList.remove('page-leaving')
})

function DomainSwitcher() {
  const { key: current } = useDomain()
  const { role } = useAuth()
  const { pathname } = useLocation()

  if (LINKS.length < 2) return null

  return (
    <div className="flex items-center rounded-full border border-slate-200 bg-slate-100 p-1">
      {LINKS.map(({ key, url }) => {
        const label = key.charAt(0).toUpperCase() + key.slice(1)
        if (key === current) {
          return (
            <span
              key={key}
              className="rounded-full bg-indigo-600 px-4 py-1.5 text-sm font-semibold text-white"
            >
              {label}
            </span>
          )
        }
        const target = `${url}${pathname}${role ? `?as=${role}` : ''}`
        return (
          <a
            key={key}
            href={target}
            onClick={(event) => leaveTo(event, target)}
            className="rounded-full px-4 py-1.5 text-sm font-semibold text-slate-600 transition hover:text-indigo-600"
          >
            {label}
          </a>
        )
      })}
    </div>
  )
}

export default DomainSwitcher
