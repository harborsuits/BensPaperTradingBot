import React from 'react'

type SimpleCardProps = {
  title: string
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}

function SimpleCardBase({ title, action, children, className = '' }: SimpleCardProps) {
  return (
    <div className={`rounded-2xl border border-white/10 bg-white/5 p-4 ${className}`}>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold opacity-80 truncate pr-2">{title}</h3>
        {action ? (
          <div className="text-xs opacity-70 hover:opacity-100">{action}</div>
        ) : null}
      </div>
      {children}
    </div>
  )
}

export default SimpleCardBase
export const SimpleCard = SimpleCardBase


