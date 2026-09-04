import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from './AuthContext.jsx'

function ProtectedRoute({ allowedRoles, children }) {
  const location = useLocation()
  const { isAuthenticated, role } = useAuth()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  if (allowedRoles?.length && !allowedRoles.includes(role)) {
    return <Navigate to="/dashboard" replace />
  }

  return children
}

export default ProtectedRoute
