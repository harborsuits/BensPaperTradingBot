import React from 'react'

type SimpleCardProps = {
  title: string
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}

function SimpleCardBase({ title, action, children, className = '' }: SimpleCardProps) {
  return (
    <div className={`card ${className}`}>
      <div className="card-header">
        <h3 className="card-title truncate pr-2">{title}</h3>
        {action ? (
          <div className="text-sm opacity-80 hover:opacity-100 transition-opacity flex-shrink-0">
            {action}
          </div>
        ) : null}
      </div>
      <div className="card-content">
        {children}
      </div>
    </div>
  )
}

export default SimpleCardBase
export const SimpleCard = SimpleCardBase


