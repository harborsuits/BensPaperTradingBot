import React from 'react';

interface Alert {
  id: string;
  type: 'warning' | 'error' | 'info';
  title: string;
  message: string;
  time: string;
}

export function AlertsPanel() {
  // Mock alerts
  const alerts: Alert[] = [
    {
      id: '1',
      type: 'warning',
      title: 'Margin Warning',
      message: 'Account margin usage is approaching 50%. Consider closing some positions.',
      time: '10 minutes ago'
    },
    {
      id: '2',
      type: 'error',
      title: 'Slippage Alert',
      message: 'High slippage detected on AAPL order: 2.3% above limit price.',
      time: '25 minutes ago'
    },
    {
      id: '3',
      type: 'info',
      title: 'Volatility Notice',
      message: 'Market volatility has increased by 15% in the last hour.',
      time: '45 minutes ago'
    }
  ];

  return (
    <div className="alerts-panel">
      {alerts.length === 0 ? (
        <div className="text-muted-foreground text-center py-8">
          No active alerts
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.map(alert => (
            <div 
              key={alert.id}
              className={`p-3 rounded border ${
                alert.type === 'warning' 
                  ? 'bg-yellow-500/10 border-yellow-500/50 text-yellow-500' 
                  : alert.type === 'error'
                    ? 'bg-red-500/10 border-red-500/50 text-red-500'
                    : 'bg-blue-500/10 border-blue-500/50 text-blue-500'
              }`}
            >
              <div className="flex justify-between items-start">
                <h3 className="font-medium">{alert.title}</h3>
                <span className="text-xs text-muted-foreground">{alert.time}</span>
              </div>
              <p className="text-sm mt-1 text-white">{alert.message}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 