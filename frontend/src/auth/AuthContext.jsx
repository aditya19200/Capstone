import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { setAuthRole, setAuthToken, setAuthUserId } from '../api/axiosClient.js'

const AuthContext = createContext(null)

const parseJwtPayload = (token) => {
  try {
    const [, payload] = token.split('.')
    return JSON.parse(atob(payload))
  } catch {
    return null
  }
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null)
  const [role, setRole] = useState(null)
  const [userId, setUserId] = useState(null)

  const login = (nextToken) => {
    const payload = parseJwtPayload(nextToken)
    const nextRole = payload?.role || payload?.user_role || payload?.app_metadata?.role || null
    const nextUserId = payload?.sub || payload?.user_id || payload?.id || null

    setToken(nextToken)
    setRole(nextRole)
    setUserId(nextUserId)
  }

  const mockLogin = (mockRole = "annotator") => {
  const fakeToken = "mock.token.value"

  setToken(fakeToken)
  setRole(mockRole)
  setUserId('mock-user-1')
  }

  const logout = () => {
    setToken(null)
    setRole(null)
    setUserId(null)
  }

  useEffect(() => {
    setAuthToken(token)
    setAuthRole(role)
    setAuthUserId(userId)
  }, [token, role, userId])

  const value = useMemo(
    () => ({
      isAuthenticated: Boolean(token),
      login,
      logout,
      role,
      token,
      userId,
      mockLogin, // 👈 ADD THIS
    }),
    [role, token, userId],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }

  return context
}
