import { useAuth } from '../../auth/AuthContext.jsx'

const roleLabels = {
  admin: 'Admin',
  annotator: 'Annotator',
  reviewer: 'Reviewer',
}

function Navbar() {
  const { logout, role } = useAuth()
  const displayRole = roleLabels[role] || 'Mock User'

  return (
    <header className="flex h-20 items-center justify-between border-b border-slate-200 bg-white px-6 shadow-sm">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-indigo-600">
          XAI Legal Workflow
        </p>
        <h1 className="text-xl font-semibold text-slate-800">Legal Annotation System</h1>
      </div>

      <div className="flex items-center gap-4">
        <div className="hidden rounded-full bg-indigo-100 px-4 py-2 text-sm font-medium text-indigo-700 sm:block">
          Role: {displayRole}
        </div>
        <button
          type="button"
          onClick={logout}
          className="btn-secondary"
        >
          Logout
        </button>
      </div>
    </header>
  )
}

export default Navbar
