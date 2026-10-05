import { createBrowserRouter, RouterProvider, Link, useParams } from 'react-router-dom'
import { Layout } from './components/Layout'
import { EmptyState } from './components/EmptyState'
import { DatasetsPage } from './pages/DatasetsPage'
import { DatasetDetailPage } from './pages/DatasetDetailPage'
import { SamplePage } from './pages/SamplePage'

/** Remount the viewer per sample so polling/re-run state resets cleanly. */
function SampleRoute() {
  const { id } = useParams()
  return <SamplePage key={id} />
}

function NotFound() {
  return (
    <div className="page">
      <div className="card">
        <EmptyState
          icon="search"
          title="Page not found"
          actions={
            <Link className="btn" to="/">
              Back to datasets
            </Link>
          }
        >
          The page you were looking for doesn’t exist.
        </EmptyState>
      </div>
    </div>
  )
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <DatasetsPage /> },
      { path: '/datasets/:id', element: <DatasetDetailPage /> },
      { path: '/samples/:id', element: <SampleRoute /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
