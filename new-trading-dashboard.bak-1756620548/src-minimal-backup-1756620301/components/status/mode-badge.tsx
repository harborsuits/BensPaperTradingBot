import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { portfolioApi } from '@/services/api'

function chipClass(kind: 'live' | 'paper' | 'unknown' | 'stop'): string {
  switch (kind) {
    case 'live':
      return 'bg-green-100 text-green-700'
    case 'paper':
      return 'bg-blue-100 text-blue-700'
    case 'stop':
      return 'bg-red-100 text-red-700'
    default:
      return 'bg-gray-200 text-gray-700'
  }
}

export default function ModeBadge() {
  const { data } = useQuery({
    queryKey: ['safety-status'],
    queryFn: () => portfolioApi.getSafetyStatus(),
    refetchInterval: 5000,
    staleTime: 3000,
  })

  const payload = (data as any)?.data ?? {}
  const rawMode: string = (payload?.mode || payload?.trading_mode || '').toString().toLowerCase()
  const emergency: boolean = Boolean(
    payload?.emergency_stop ?? payload?.emergency_stop_active ?? payload?.emergency
  )

  const kind: 'live' | 'paper' | 'unknown' | 'stop' = emergency
    ? 'stop'
    : rawMode === 'live'
    ? 'live'
    : rawMode === 'paper'
    ? 'paper'
    : 'unknown'

  const label = emergency
    ? 'STOP'
    : rawMode === 'live'
    ? 'LIVE'
    : rawMode === 'paper'
    ? 'PAPER'
    : 'MODE —'

  return (
    <span className={`text-xs px-2 py-0.5 rounded ${chipClass(kind)}`}>Mode: {label}</span>
  )
}


