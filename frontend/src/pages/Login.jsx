import { useAuth } from "../auth/AuthContext"
import { useNavigate } from "react-router-dom"

function LoginPage() {
  const { mockLogin } = useAuth()
  const navigate = useNavigate()

  const handleLogin = () => {
    mockLogin("annotator") // change role if needed
    navigate("/dashboard")
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-6">
      <div className="dashboard-card w-full max-w-sm">
        <h1 className="mb-6 text-center text-2xl font-semibold text-slate-800">Login</h1>

        <button
          onClick={handleLogin}
          className="btn-primary w-full py-3"
        >
          Mock Login
        </button>
      </div>
    </div>
  )
}

export default LoginPage
