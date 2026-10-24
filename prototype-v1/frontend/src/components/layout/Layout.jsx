import { Outlet, useLocation } from 'react-router-dom'
import Navbar from './Navbar.jsx'
import Sidebar from './Sidebar.jsx'

function Layout() {
  const { pathname } = useLocation()

  return (
    <div className="flex h-screen bg-slate-50">
      <Sidebar />

      <div className="flex flex-1 flex-col">
        <Navbar />

        <main className="flex-1 overflow-y-auto p-6">
          {/* Keyed by path so each page fades in; the sidebar and navbar stay put. */}
          <div key={pathname} className="page-fade mx-auto w-full max-w-7xl">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

export default Layout
