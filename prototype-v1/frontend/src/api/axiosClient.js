import axios from 'axios'

let authToken = null
// The current backend checks role via X-Role / X-User-Id headers (mocked
// auth — see backend/routers/annotate.py), not the Authorization bearer
// token. Both are sent; swap this once real JWT-based auth lands server-side.
let authRole = null
let authUserId = null

export const setAuthToken = (token) => {
  authToken = token
}

export const setAuthRole = (role) => {
  authRole = role
}

export const setAuthUserId = (userId) => {
  authUserId = userId
}

const axiosClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
})

axiosClient.interceptors.request.use(
  (config) => {
    if (authToken) {
      config.headers.Authorization = `Bearer ${authToken}`
    }
    if (authRole) {
      config.headers['X-Role'] = authRole
    }
    if (authUserId) {
      config.headers['X-User-Id'] = authUserId
    }

    return config
  },
  (error) => Promise.reject(error),
)

export default axiosClient
