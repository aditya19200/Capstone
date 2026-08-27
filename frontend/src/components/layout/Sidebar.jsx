import { NavLink } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext.jsx'

const navigationByRole = {
  admin: [
    { label: 'Dashboard', to: '/dashboard' },
    { label: 'Admin', to: '/admin' },
    { label: 'Metrics', to: '/metrics' },
    { label: 'Ontology', to: '/ontology' },
    { label: 'Retrain', to: '/retrain' },
  ],
  annotator: [
    { label: 'Dashboard', to: '/dashboard' },
    { label: 'Annotate', to: '/annotate' },
    { label: 'Predictions', to: '/predictions' },
    { label: 'Ontology', to: '/ontology' },
  ],
  reviewer: [
    { label: 'Dashboard', to: '/dashboard' },
    { label: 'Review', to: '/review' },
    { label: 'Conflicts', to: '/conflicts' },
  ],
}

function Sidebar() {
  const { role } = useAuth()
  const navItems = navigationByRole[role] || navigationByRole.annotator

  return (
    <aside className="hidden h-screen w-64 flex-shrink-0 bg-slate-900 text-slate-300 lg:block">
      <div className="flex h-full flex-col px-5 py-6">
        <div className="border-b border-slate-800 pb-5">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-indigo-300">
            Workspace
          </p>
          <p className="mt-2 text-sm text-slate-400">
            Navigate role-based legal annotation workflows.
          </p>
        </div>

        <nav className="mt-6 flex flex-1 flex-col gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                [
                  'rounded-xl px-4 py-3 text-sm font-medium transition',
                  isActive
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-300 hover:bg-slate-800 hover:text-white',
                ].join(' ')
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </aside>
  )
}

export default Sidebar
