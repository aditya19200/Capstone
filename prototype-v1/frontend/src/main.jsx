import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.jsx'
import { AuthProvider } from './auth/AuthContext.jsx'
import DomainProvider from './domain/DomainProvider.jsx'
import { playArrival } from './domain/switchTransition.js'

const queryClient = new QueryClient()

// Before the first render, so the arriving half of a domain switch covers
// the page from its first frame.
playArrival()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <DomainProvider>
          <App />
        </DomainProvider>
      </AuthProvider>
    </QueryClientProvider>
  </StrictMode>,
)
