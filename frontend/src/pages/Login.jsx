import { useAuth } from "../auth/AuthContext"
import { useNavigate } from "react-router-dom"

// Temporary role picker until Supabase auth is wired up — lets any role's
// screens be reached without a real login flow.
const ROLES = [
  { role: "annotator", label: "Log in as Annotator" },
  { role: "reviewer", label: "Log in as Reviewer" },
  { role: "admin", label: "Log in as Admin" },
]

function LoginPage() {
  const { mockLogin } = useAuth()
  const navigate = useNavigate()

  const handleLogin = (role) => {
    mockLogin(role)
    navigate("/dashboard")
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-6">
      <div className="dashboard-card w-full max-w-sm">
        <h1 className="mb-6 text-center text-2xl font-semibold text-slate-800">Login</h1>

        <div className="flex flex-col gap-3">
          {ROLES.map(({ role, label }) => (
            <button
              key={role}
              onClick={() => handleLogin(role)}
              className="btn-primary w-full py-3"
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default LoginPage
