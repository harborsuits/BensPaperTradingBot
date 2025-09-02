import React from 'react'

interface RightSidebarProps {}

const RightSidebar: React.FC<RightSidebarProps> = () => {
  return (
    <div className="right-sidebar w-72 h-full border-l border-border bg-card p-4 flex flex-col">
      <div className="sidebar-header mb-4">
        <h2 className="text-xl font-bold">Dashboard | Backtesting</h2>
      </div>
      
      {/* P/L Total */}
      <div className="stat-box mb-4 p-3 bg-muted rounded-md">
        <h3 className="text-sm font-medium mb-1">Profit / Loss Total</h3>
        <p className="text-2xl font-bold text-primary">+$12,458.32</p>
        <p className="text-xs text-muted-foreground">+18.7% all time</p>
      </div>
      
      {/* P/L for the day */}
      <div className="stat-box mb-4 p-3 bg-muted rounded-md">
        <h3 className="text-sm font-medium mb-1">P/L for the day</h3>
        <p className="text-2xl font-bold text-green-500">+$423.65</p>
        <p className="text-xs text-muted-foreground">+0.63% today</p>
      </div>
      
      {/* Open positions */}
      <div className="stat-box mb-4 p-3 bg-muted rounded-md">
        <h3 className="text-sm font-medium mb-1">Open positions</h3>
        <div className="positions-list text-sm space-y-2 mt-2">
          <div className="position flex justify-between">
            <span>AAPL</span>
            <span className="text-green-500">+2.4%</span>
          </div>
          <div className="position flex justify-between">
            <span>MSFT</span>
            <span className="text-red-500">-0.8%</span>
          </div>
          <div className="position flex justify-between">
            <span>NVDA</span>
            <span className="text-green-500">+5.2%</span>
          </div>
        </div>
      </div>
      
      {/* Win/rate */}
      <div className="stat-box mb-4 p-3 bg-muted rounded-md">
        <h3 className="text-sm font-medium mb-1">Win/rate</h3>
        <p className="text-2xl font-bold">72%</p>
        <p className="text-xs text-muted-foreground">36 wins / 14 losses</p>
      </div>
      
      {/* Exposure */}
      <div className="stat-box mb-4 p-3 bg-muted rounded-md">
        <h3 className="text-sm font-medium mb-1">Exposure</h3>
        <div className="flex justify-between items-center">
          <p className="text-xl font-bold">78%</p>
          <div className="w-2/3 h-2 bg-muted-foreground/20 rounded-full overflow-hidden">
            <div className="h-full bg-primary rounded-full" style={{ width: '78%' }}></div>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-1">22% cash available</p>
      </div>
      
      {/* API Bal $ */}
      <div className="stat-box mb-4 p-3 bg-muted rounded-md">
        <h3 className="text-sm font-medium mb-1">API Bal $</h3>
        <p className="text-2xl font-bold">$78,932.45</p>
        <p className="text-xs text-muted-foreground">Updated 5 min ago</p>
      </div>
    </div>
  )
}

export default RightSidebar 