import ConfidenceBadge from '../prediction/ConfidenceBadge.jsx'

const PAGE_SIZE = 10

function BatchItemsTable({
  items,
  page,
  totalItems,
  isLoading,
  onPageChange,
  onExplain,
  onConcepts,
  exportUrl,
}) {
  const maxPage = Math.max(1, Math.ceil((totalItems || 0) / PAGE_SIZE))

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between border-b border-slate-200 pb-5">
        <div>
          <p className="section-kicker">Batch Results</p>
          <h3 className="mt-2 text-xl font-semibold text-slate-900">Classified documents</h3>
        </div>
        <div className="flex items-center gap-3">
          {exportUrl ? (
            <a href={exportUrl} download className="btn-secondary">
              Export CSV
            </a>
          ) : null}
          <button
            type="button"
            className="btn-secondary"
            disabled={page <= 1 || isLoading}
            onClick={() => onPageChange(page - 1)}
          >
            Prev
          </button>
          <span className="text-sm text-slate-500">
            Page {page} of {maxPage}
          </span>
          <button
            type="button"
            className="btn-secondary"
            disabled={page >= maxPage || isLoading}
            onClick={() => onPageChange(page + 1)}
          >
            Next
          </button>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-100">
            <tr>
              {['Text', 'Predicted Label', 'Confidence', ''].map((heading) => (
                <th
                  key={heading}
                  scope="col"
                  className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.16em] text-slate-500"
                >
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 bg-white">
            {items.map((item) => (
              <tr key={item.id}>
                <td className="max-w-sm px-4 py-4 text-sm text-slate-700">
                  <p className="truncate">{item.text_content}</p>
                </td>
                <td className="px-4 py-4 text-sm font-medium text-slate-800">
                  {item.predicted_label}
                </td>
                <td className="px-4 py-4">
                  <ConfidenceBadge confidence={item.confidence} />
                </td>
                <td className="px-4 py-4 text-right">
                  <div className="flex justify-end gap-2">
                    <button type="button" className="btn-secondary" onClick={() => onConcepts(item)}>
                      Concepts
                    </button>
                    <button type="button" className="btn-secondary" onClick={() => onExplain(item)}>
                      Explain
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default BatchItemsTable
