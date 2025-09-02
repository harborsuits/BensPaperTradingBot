// NEW: created because no existing equivalent was found; composes existing status parts.
import BackendStatus from '@/components/status/backend-status'
import WsStatus from '@/components/status/ws-status'
import DataAgeBadge from '@/components/status/data-age-badge'
import ModeBadge from '@/components/status/mode-badge'

export default function HeaderBar() {
  return (
    <div className="w-full flex items-center justify-between p-3 border-b bg-white">
      <div className="text-xl font-semibold">BenBot Dashboard</div>
      <div className="flex items-center gap-3 text-sm">
        <ModeBadge />
        <BackendStatus />
        <WsStatus />
        <DataAgeBadge />
      </div>
    </div>
  )
}


