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

// The header's domain switcher moves between two instances on different ports,
// which are different origins, so nothing in this tab's memory reaches the
// other. It passes the role along as ?as=<role> instead, and the receiving tab
// starts signed in as it. Mock auth only — there is no credential here to leak.
const SWITCHABLE_ROLES = ['annotator', 'reviewer', 'admin']

function roleFromUrl() {
  if (typeof window === 'undefined') return null
  const as = new URLSearchParams(window.location.search).get('as')
  return SWITCHABLE_ROLES.includes(as) ? as : null
}

export function AuthProvider({ children }) {
  const [carriedRole] = useState(() => {
    const carried = roleFromUrl()
    if (carried) {
      // Synchronously, not in the effect below: React runs children's effects
      // before a parent's, so the page we land on would otherwise send its
      // first request before the role header exists and get a 403.
      // Idempotent, so StrictMode calling this twice is harmless.
      setAuthToken('mock.token.value')
      setAuthRole(carried)
      setAuthUserId('mock-user-1')
    }
    return carried
  })
  const [token, setToken] = useState(carriedRole ? 'mock.token.value' : null)
  const [role, setRole] = useState(carriedRole)
  const [userId, setUserId] = useState(carriedRole ? 'mock-user-1' : null)

  // Drop ?as= once read, so a reload or a copied link does not re-assert it.
  useEffect(() => {
    if (!carriedRole) return
    const url = new URL(window.location.href)
    url.searchParams.delete('as')
    window.history.replaceState(null, '', url.pathname + url.search + url.hash)
  }, [carriedRole])

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
