import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { setAuthRole, setAuthToken, setAuthUserId } from '../api/axiosClient.js'

const AuthContext = createContext(null)

const parseJwtPayload = (token) => {
  try {
    const [, payload] = token.split('.')
    if (!payload) return null

    const normalized = payload.replace(/-/g, '+').replace(/_/g, '/')
    const padded = normalized.padEnd(normalized.length + ((4 - (normalized.length % 4)) % 4), '=')

    return JSON.parse(atob(padded))
  } catch {
    return null
  }
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null)
  const [role, setRole] = useState(null)
  const [userId, setUserId] = useState(null)

  // useCallback with an empty deps array — these only call the setState
  // setters (guaranteed stable by React), so they never need to change
  // identity. That's what lets them sit correctly in the useMemo deps
  // below without invalidating the memo on every render.
  const login = useCallback((nextToken) => {
    const payload = parseJwtPayload(nextToken)
    const nextRole = payload?.role || payload?.user_role || payload?.app_metadata?.role || null
    const nextUserId = payload?.sub || payload?.user_id || payload?.id || null

    setToken(nextToken)
    setRole(nextRole)
    setUserId(nextUserId)
  }, [])

  const mockLogin = useCallback((mockRole = "annotator") => {
    const fakeToken = "mock.token.value"

    setToken(fakeToken)
    setRole(mockRole)
    setUserId('mock-user-1')
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setRole(null)
    setUserId(null)
  }, [])

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
      mockLogin,
    }),
    [role, token, userId, login, logout, mockLogin],
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
