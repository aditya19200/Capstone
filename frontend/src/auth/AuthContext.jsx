import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { setAuthToken } from '../api/axiosClient.js'

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

  const login = (nextToken) => {
    const payload = parseJwtPayload(nextToken)
    const nextRole = payload?.role || payload?.user_role || payload?.app_metadata?.role || null

    setToken(nextToken)
    setRole(nextRole)
  }

  const mockLogin = (mockRole = "annotator") => {
  const fakeToken = "mock.token.value"

  setToken(fakeToken)
  setRole(mockRole)
  }

  const logout = () => {
    setToken(null)
    setRole(null)
  }

  useEffect(() => {
    setAuthToken(token)
  }, [token])

  const value = useMemo(
    () => ({
      isAuthenticated: Boolean(token),
      login,
      logout,
      role,
      token,
      mockLogin, // 👈 ADD THIS
    }),
    [role, token],
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
