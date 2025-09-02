import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Bell, 
  CheckCircle, 
  XCircle, 
  RefreshCw, 
  Filter, 
  AlarmClock 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { 
  RiskAlert, 
  RiskAlertSeverity,
  getAlerts,
  acknowledgeAlert, 
  clearAcknowledgedAlerts,
  subscribeToAlerts
} from '@/services/riskAlerts';
import { formatDistanceToNow } from 'date-fns';

interface RiskAlertsPanelProps {
  className?: string;
}

const RiskAlertsPanel: React.FC<RiskAlertsPanelProps> = ({ className }) => {
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [showAcknowledged, setShowAcknowledged] = useState(false);
  
  useEffect(() => {
    // Get initial alerts
    setAlerts(getAlerts());
    
    // Subscribe to alert updates
    const unsubscribe = subscribeToAlerts((updatedAlerts) => {
      setAlerts(updatedAlerts);
    });
    
    return () => {
      unsubscribe();
    };
  }, []);
  
  // Apply filters to alerts
  const filteredAlerts = alerts.filter(alert => {
    // Filter by severity
    if (filter !== 'all' && alert.severity !== filter) {
      return false;
    }
    
    // Filter by acknowledged status
    if (!showAcknowledged && alert.acknowledged) {
      return false;
    }
    
    return true;
  });
  
  // Get alert type badge color
  const getSeverityColor = (severity: RiskAlertSeverity) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-red-600';
      case 'HIGH':
        return 'bg-red-500';
      case 'MEDIUM':
        return 'bg-amber-500';
      case 'LOW':
        return 'bg-blue-500';
      default:
        return 'bg-slate-500';
    }
  };
  
  const handleAcknowledge = (alertId: string) => {
    acknowledgeAlert(alertId);
  };
  
  const handleClearAcknowledged = () => {
    clearAcknowledgedAlerts();
  };
  
  return (
    <Card className={className}>
      <CardHeader className="relative">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">Risk Alerts</CardTitle>
          <div className="flex items-center gap-2">
            <div className="flex items-center space-x-1 bg-muted/20 rounded-lg py-1 px-2">
              <Filter size={14} className="text-muted-foreground" />
              <select 
                className="bg-transparent border-none text-xs focus:outline-none focus:ring-0"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                aria-label="Filter alerts by severity"
                title="Filter alerts by severity"
              >
                <option value="all">All Severities</option>
                <option value="CRITICAL">Critical</option>
                <option value="HIGH">High</option>
                <option value="MEDIUM">Medium</option>
                <option value="LOW">Low</option>
              </select>
            </div>
            <Button
              variant="ghost" 
              size="icon"
              className="p-1 h-8 w-8"
              onClick={() => setShowAcknowledged(!showAcknowledged)}
              title={showAcknowledged ? "Hide acknowledged alerts" : "Show acknowledged alerts"}
            >
              {showAcknowledged ? <CheckCircle size={16} /> : <XCircle size={16} />}
            </Button>
            <Button
              variant="ghost" 
              size="icon"
              className="p-1 h-8 w-8"
              onClick={handleClearAcknowledged}
              title="Clear acknowledged alerts"
            >
              <XCircle size={16} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {filteredAlerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <Bell size={32} className="mb-2" />
              <p className="text-sm">{filter === 'all' ? 'No alerts' : `No ${filter.toLowerCase()} alerts`}</p>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredAlerts.map((alert) => (
                <div 
                  key={alert.id} 
                  className={`p-3 border rounded-lg ${alert.acknowledged ? 'bg-muted/10 border-muted/20' : 'bg-card border-border'}`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex items-start gap-2">
                      <AlertTriangle 
                        size={18} 
                        className={alert.severity === 'CRITICAL' ? 'text-red-600' : 
                                    alert.severity === 'HIGH' ? 'text-red-500' : 
                                    alert.severity === 'MEDIUM' ? 'text-amber-500' : 
                                    'text-blue-500'} 
                      />
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium text-sm">{alert.type}</h3>
                          <Badge className={`text-xs px-1.5 py-0 ${getSeverityColor(alert.severity)}`}>
                            {alert.severity}
                          </Badge>
                        </div>
                        <p className="text-sm mt-1">{alert.message}</p>
                        <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                          <span className="flex items-center">
                            <AlarmClock size={12} className="mr-1" />
                            {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                          </span>
                          <span>Source: {alert.source}</span>
                        </div>
                      </div>
                    </div>
                    {!alert.acknowledged && (
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="text-xs px-2 py-1 h-auto"
                        onClick={() => handleAcknowledge(alert.id)}
                      >
                        Acknowledge
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default RiskAlertsPanel;
