import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import ProtectedRoute from './auth/ProtectedRoute.jsx'
import Layout from './components/layout/Layout.jsx'
import AdminPage from './pages/Admin.jsx'
import AnnotatePage from './pages/Annotate.jsx'
import ConflictsPage from './pages/Conflicts.jsx'
import DashboardPage from './pages/Dashboard.jsx'
import LoginPage from './pages/Login.jsx'
import MetricsPage from './pages/Metrics.jsx'
import OntologyPage from './pages/Ontology.jsx'
import RegisterPage from './pages/Register.jsx'
import RetrainPage from './pages/Retrain.jsx'
import ReviewPage from './pages/Review.jsx'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
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
            path="/ontology"
            element={
              <ProtectedRoute allowedRoles={['admin', 'annotator']}>
                <OntologyPage />
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
        </Route>

        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
