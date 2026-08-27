import { useState } from 'react'
import AnnotationActions from '../components/annotation/AnnotationActions.jsx'
import ExplanationPanel from '../components/explanation/ExplanationPanel.jsx'
import KnowledgeGraphPanel from '../components/ontology/KnowledgeGraphPanel.jsx'
import PredictionCard from '../components/prediction/PredictionCard.jsx'
import StatusBadge from '../components/prediction/StatusBadge.jsx'


const mockDocuments = [
 {
   id: 'DOC-2041',
   title: 'Vendor Agreement Review',
   uploadedAt: '2026-04-02',
   predictedLabel: 'Contract Law',
   confidenceScore: 0.87,
   routingDecision: 'AUTO_ACCEPT',
   summary:
     'The model identified contractual obligations, indemnity language, and termination clauses that strongly align with a commercial agreement context.',
   shapValues: [
     { token: 'The', value: 0.04 },
     { token: 'agreement', value: 0.92 },
     { token: 'between', value: 0.05 },
     { token: 'the', value: 0.03 },
     { token: 'parties', value: 0.61 },
     { token: 'creates', value: 0.14 },
     { token: 'binding', value: 0.82 },
     { token: 'obligations', value: 0.88 },
     { token: 'for', value: 0.02 },
     { token: 'delivery', value: 0.22 },
     { token: 'termination', value: 0.74 },
     { token: 'damages', value: -0.27 },
   ],
   concepts: [
     { name: 'Offer and Acceptance', relation: 'Foundational concept' },
     { name: 'Consideration', relation: 'Linked doctrine' },
     { name: 'Termination Clause', relation: 'Referenced clause' },
   ],
 },
 {
   id: 'DOC-2042',
   title: 'Constitutional Writ Petition',
   uploadedAt: '2026-04-01',
   predictedLabel: 'Constitutional Law',
   confidenceScore: 0.73,
   routingDecision: 'NEEDS_EXPLANATION',
   summary:
     'The language references fundamental rights, state action, and a request for judicial review, which signal a constitutional dispute.',
   shapValues: [
     { token: 'The', value: 0.03 },
     { token: 'petitioner', value: 0.21 },
     { token: 'seeks', value: 0.18 },
     { token: 'writ', value: 0.84 },
     { token: 'relief', value: 0.19 },
     { token: 'against', value: 0.04 },
     { token: 'state', value: 0.79 },
     { token: 'action', value: 0.43 },
     { token: 'violating', value: 0.52 },
     { token: 'fundamental', value: 0.91 },
     { token: 'rights', value: 0.89 },
     { token: 'damages', value: -0.24 },
   ],
   concepts: [
     { name: 'Judicial Review', relation: 'Primary doctrine' },
     { name: 'Fundamental Rights', relation: 'Constitutional issue' },
     { name: 'State Action', relation: 'Linked principle' },
   ],
 },
 {
   id: 'DOC-2043',
   title: 'Employment Dismissal Notice',
   uploadedAt: '2026-03-31',
   predictedLabel: 'Labour & Employment Law',
   confidenceScore: 0.54,
   routingDecision: 'ROUTE_TO_REVIEWER',
   summary:
     'Employment-specific terminology such as termination, wages, and workplace obligations indicate a labour dispute, though the confidence is lower.',
   shapValues: [
     { token: 'Employee', value: 0.65 },
     { token: 'termination', value: 0.57 },
     { token: 'notice', value: 0.28 },
     { token: 'wages', value: 0.76 },
     { token: 'disciplinary', value: 0.31 },
     { token: 'hearing', value: -0.19 },
     { token: 'workplace', value: 0.62 },
     { token: 'misconduct', value: 0.22 },
     { token: 'union', value: 0.48 },
     { token: 'benefits', value: 0.51 },
   ],
   concepts: [
     { name: 'Termination Procedure', relation: 'Related process' },
     { name: 'Wage Protection', relation: 'Connected right' },
     { name: 'Industrial Relations', relation: 'Linked domain' },
   ],
 },
]


const actionLabels = {
 accept: 'Prediction accepted.',
 modify: 'Prediction sent for label modification.',
 flag: 'Prediction flagged as uncertain.',
}


function PredictionsPage() {
 const [selectedDocument, setSelectedDocument] = useState(null)
 const [lastAction, setLastAction] = useState('')


 return (
   <section className="relative">
     <div className="space-y-6">
       <div className="max-w-3xl">
         <p className="text-sm font-semibold uppercase tracking-[0.24em] text-indigo-600">
           Predictions Workspace
         </p>
         <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
           Batch prediction review
         </h2>
         <p className="mt-3 text-sm leading-6 text-slate-600 sm:text-base">
           Inspect uploaded documents, open a detailed explanation panel, and decide how each
           prediction should move through the annotation workflow.
         </p>
       </div>


       <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
         <div className="flex items-center justify-between border-b border-slate-200 px-6 py-5">
           <div>
             <p className="section-kicker">Documents</p>
             <h3 className="mt-2 text-xl font-semibold text-slate-900">
               Uploaded legal records
             </h3>
           </div>
           <div className="rounded-full bg-indigo-100 px-4 py-2 text-sm font-medium text-indigo-700">
             {mockDocuments.length} Documents
           </div>
         </div>


         {!selectedDocument ? (
           <div className="border-b border-slate-200 bg-slate-50 px-6 py-4 text-sm text-slate-600">
             Select a document row to open its prediction details panel.
           </div>
         ) : null}


         <div className="overflow-x-auto">
           <table className="min-w-full divide-y divide-slate-200">
             <thead className="bg-slate-100">
               <tr>
                 {['Document', 'Uploaded', 'Prediction', 'Confidence', 'Status'].map((heading) => (
                   <th
                     key={heading}
                     scope="col"
                     className="px-6 py-4 text-left text-xs font-semibold uppercase tracking-[0.16em] text-slate-500"
                   >
                     {heading}
                   </th>
                 ))}
               </tr>
             </thead>
             <tbody className="divide-y divide-slate-200 bg-white">
               {mockDocuments.map((document) => (
                 <tr
                   key={document.id}
                   onClick={() => {
                     setSelectedDocument(document)
                     setLastAction('')
                   }}
                   className={[
                     'cursor-pointer transition hover:bg-slate-50',
                     selectedDocument?.id === document.id ? 'bg-slate-50' : '',
                   ].join(' ')}
                 >
                   <td className="px-6 py-4">
                     <p className="font-semibold text-slate-900">{document.title}</p>
                     <p className="mt-1 text-sm text-slate-500">{document.id}</p>
                   </td>
                   <td className="px-6 py-4 text-sm text-slate-600">{document.uploadedAt}</td>
                   <td className="px-6 py-4 text-sm font-medium text-slate-800">
                     {document.predictedLabel}
                   </td>
                   <td className="px-6 py-4 text-sm text-slate-600">
                     {Math.round(document.confidenceScore * 100)}%
                   </td>
                   <td className="px-6 py-4">
                     <StatusBadge routingDecision={document.routingDecision} />
                   </td>
                 </tr>
               ))}
             </tbody>
           </table>
         </div>
       </div>
     </div>


     {selectedDocument ? (
       <>
         <button
           type="button"
           aria-label="Close details panel backdrop"
           className="fixed inset-0 z-30 bg-slate-950/30 backdrop-blur-sm"
           onClick={() => setSelectedDocument(null)}
         />


         <aside className="fixed bottom-0 right-0 top-20 z-40 w-full max-w-2xl overflow-y-auto border-l border-slate-200 bg-slate-50 shadow-2xl">
           <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/95 px-6 py-5 backdrop-blur">
             <div>
               <p className="section-kicker">Document Details</p>
               <h3 className="mt-2 text-xl font-semibold text-slate-900">
                 {selectedDocument.title}
               </h3>
             </div>
             <button
               type="button"
               onClick={() => setSelectedDocument(null)}
               className="btn-secondary"
             >
               Close
             </button>
           </div>


           <div className="space-y-6 p-6">
             <PredictionCard
               predictedLabel={selectedDocument.predictedLabel}
               confidenceScore={selectedDocument.confidenceScore}
               routingDecision={selectedDocument.routingDecision}
             />


             <ExplanationPanel
               summary={selectedDocument.summary}
               shapValues={selectedDocument.shapValues}
             />


             <KnowledgeGraphPanel concepts={selectedDocument.concepts} />


             <AnnotationActions onAction={(action) => setLastAction(actionLabels[action])} />


             {lastAction ? (
               <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 shadow-sm">
                 {lastAction}
               </div>
             ) : null}
           </div>
         </aside>
       </>
     ) : null}
   </section>
 )
}


export default PredictionsPage



