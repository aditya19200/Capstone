// The legal <-> medical switch. The domains are separate pages on separate
// ports, so the page being left only fades out, then navigates with
// ?switched=<domain>&sx=&sy= (where the click was, as fractions of the
// window). The page arriving plays the whole thing: a panel in its colour
// grows from that point, its name and icon come up, the icon plays, and the
// panel lifts away to show the page.
//
// Purely decorative. Reduced motion, or a modified click, skips it.

const THEMES = {
  legal: {
    from: '#4f46e5',
    to: '#1e1b4b',
    kicker: 'Indian court judgments',
    // Scales of justice; the beam tips once.
    icon: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="3"
             stroke-linecap="round" stroke-linejoin="round">
             <path d="M32 10v42M22 54h20"/>
             <g class="dt-beam">
               <path d="M12 18h40"/>
               <path d="M16 18l-7 16h14zM48 18l-7 16h14z"/>
               <path d="M5 34a11 5 0 0 0 22 0M37 34a11 5 0 0 0 22 0" stroke-width="2.5"/>
             </g>
             <circle cx="32" cy="10" r="3" fill="currentColor"/>
           </svg>`,
  },
  medical: {
    from: '#0d9488',
    to: '#042f2e',
    kicker: 'Clinical notes',
    // A heartbeat trace; a bright segment sweeps along it.
    icon: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="3.5"
             stroke-linecap="round" stroke-linejoin="round">
             <path class="dt-pulse-base" pathLength="100"
                   d="M2 34h14l5-12 8 26 7-30 6 22 4-6h16"/>
             <path class="dt-pulse-trace" pathLength="100"
                   d="M2 34h14l5-12 8 26 7-30 6 22 4-6h16"/>
           </svg>`,
  },
}

// Matches the #root transition on html.page-leaving in index.css.
const LEAVE_MS = 160

function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

// The page being left: fade out, then go.
export function switchTo(event, key, target) {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
  if (reducedMotion()) return
  event.preventDefault()

  const url = new URL(target)
  url.searchParams.set('switched', key)
  // A keyboard "click" reports (0, 0); the panel then grows from the middle.
  if (event.clientX || event.clientY) {
    url.searchParams.set('sx', (event.clientX / window.innerWidth).toFixed(3))
    url.searchParams.set('sy', (event.clientY / window.innerHeight).toFixed(3))
  }

  document.documentElement.classList.add('page-leaving')
  window.setTimeout(() => {
    window.location.href = url.toString()
  }, LEAVE_MS)
}

function fraction(value) {
  const n = Number.parseFloat(value)
  return Number.isFinite(n) && n >= 0 && n <= 1 ? n : 0.5
}

// The page arriving: called once from main.jsx, before React renders, so the
// overlay covers the page from its first frame.
export function playArrival() {
  const url = new URL(window.location.href)
  const key = url.searchParams.get('switched')
  if (!key) return
  const x = fraction(url.searchParams.get('sx'))
  const y = fraction(url.searchParams.get('sy'))
  for (const name of ['switched', 'sx', 'sy']) url.searchParams.delete(name)
  window.history.replaceState(null, '', url.pathname + url.search + url.hash)

  const theme = THEMES[key]
  if (!theme || reducedMotion()) return
  const label = key.charAt(0).toUpperCase() + key.slice(1)

  const overlay = document.createElement('div')
  overlay.className = 'domain-transition'
  overlay.setAttribute('aria-hidden', 'true')
  overlay.style.setProperty('--dt-from', theme.from)
  overlay.style.setProperty('--dt-to', theme.to)
  overlay.style.setProperty('--dt-x', `${x * 100}%`)
  overlay.style.setProperty('--dt-y', `${y * 100}%`)
  overlay.innerHTML = `
    <div class="dt-panel">
      <div class="dt-content">
        <div class="dt-icon">${theme.icon}</div>
        <div class="dt-label">${label}</div>
        <div class="dt-kicker">${theme.kicker}</div>
      </div>
    </div>`
  overlay.addEventListener('animationend', (e) => {
    if (e.target === overlay) overlay.remove()
  })
  // In case animationend never fires (a tab opened in the background).
  window.setTimeout(() => overlay.remove(), 3500)
  document.body.appendChild(overlay)
}

// Back from the other domain can restore this page exactly as it was left:
// faded out. Undo that whenever the page is shown again.
window.addEventListener('pageshow', () => {
  document.documentElement.classList.remove('page-leaving')
})
