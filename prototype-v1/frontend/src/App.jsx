import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import ProtectedRoute from './auth/ProtectedRoute.jsx'
import Layout from './components/layout/Layout.jsx'
import AdminPage from './pages/Admin.jsx'
import AnnotatePage from './pages/Annotate.jsx'
import ConflictsPage from './pages/Conflicts.jsx'
import DashboardPage from './pages/Dashboard.jsx'
import HomePage from './pages/Home.jsx'
import LoginPage from './pages/Login.jsx'
import MetricsPage from './pages/Metrics.jsx'
import RegisterPage from './pages/Register.jsx'
import RetrainPage from './pages/Retrain.jsx'
import ComparePage from './pages/Compare.jsx'
import ReviewPage from './pages/Review.jsx'

const PUBLIC_PATHS = ['/', '/login', '/register']

// Fades each public page in, and the workspace once on entering it (after
// signing in). Inside the workspace Layout fades the page alone, so the key
// stays fixed there and the sidebar is not re-mounted on every click.
function PageFade({ children }) {
  const { pathname } = useLocation()
  const key = PUBLIC_PATHS.includes(pathname) ? pathname : 'workspace'
  return (
    <div key={key} className="page-fade">
      {children}
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <PageFade>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          <Route
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route path="/dashboard" element={<DashboardPage />} />

            <Route
              path="/annotate"
              element={
                <ProtectedRoute allowedRoles={['annotator']}>
                  <AnnotatePage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/review"
              element={
                <ProtectedRoute allowedRoles={['reviewer']}>
                  <ReviewPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/conflicts"
              element={
                <ProtectedRoute allowedRoles={['reviewer', 'admin']}>
                  <ConflictsPage />
                </ProtectedRoute>
              }
            />

            <Route
              path="/admin"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/metrics"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <MetricsPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/retrain"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <RetrainPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/compare"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <ComparePage />
                </ProtectedRoute>
              }
            />
          </Route>

          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </PageFade>
    </BrowserRouter>
  )
}

export default App
