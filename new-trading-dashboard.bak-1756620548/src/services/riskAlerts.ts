import { toast } from 'react-hot-toast';

export type RiskAlertSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface RiskAlert {
  id: string;
  type: string;
  message: string;
  severity: RiskAlertSeverity;
  timestamp: string;
  source: string;
  details?: Record<string, any>;
  acknowledged?: boolean;
}

// Risk alert types that might come from the backend
export enum RiskAlertType {
  DRAWDOWN = 'DRAWDOWN',
  VOLATILITY = 'VOLATILITY',
  BROKER_ISSUE = 'BROKER_ISSUE',
  MARKET_EVENT = 'MARKET_EVENT',
  STRATEGY_ANOMALY = 'STRATEGY_ANOMALY',
  POSITION_RISK = 'POSITION_RISK',
  CIRCUIT_BREAKER = 'CIRCUIT_BREAKER',
  CORRELATION_SHIFT = 'CORRELATION_SHIFT',
  LIQUIDITY_ISSUE = 'LIQUIDITY_ISSUE',
}

// Global cache of received alerts
let alertsCache: RiskAlert[] = [];

// Add a new alert to the cache
export const addAlert = (alert: RiskAlert) => {
  // Check if this alert already exists (by ID)
  const existingIndex = alertsCache.findIndex(a => a.id === alert.id);
  
  if (existingIndex >= 0) {
    // Update existing alert
    alertsCache[existingIndex] = {
      ...alertsCache[existingIndex],
      ...alert
    };
  } else {
    // Add new alert
    alertsCache = [alert, ...alertsCache].slice(0, 100); // Keep only the last 100 alerts
  }
  
  // Notify subscribers
  subscribers.forEach(callback => callback(alertsCache));
  
  return alertsCache;
};

// Get all alerts
export const getAlerts = () => alertsCache;

// Mark an alert as acknowledged
export const acknowledgeAlert = (alertId: string) => {
  const alertIndex = alertsCache.findIndex(a => a.id === alertId);
  
  if (alertIndex >= 0) {
    alertsCache[alertIndex] = {
      ...alertsCache[alertIndex],
      acknowledged: true
    };
    
    // Notify subscribers
    subscribers.forEach(callback => callback(alertsCache));
    
    // Call API to acknowledge the alert on the backend
    fetch(`/api/alerts/${alertId}/acknowledge`, {
      method: 'POST'
    }).catch(err => {
      console.error('Failed to acknowledge alert on server:', err);
    });
  }
  
  return alertsCache;
};

// Clear all acknowledged alerts
export const clearAcknowledgedAlerts = () => {
  alertsCache = alertsCache.filter(alert => !alert.acknowledged);
  
  // Notify subscribers
  subscribers.forEach(callback => callback(alertsCache));
  
  return alertsCache;
};

// Subscribers for alert updates
type AlertSubscriber = (alerts: RiskAlert[]) => void;
const subscribers: AlertSubscriber[] = [];

// Subscribe to alert updates
export const subscribeToAlerts = (callback: AlertSubscriber) => {
  subscribers.push(callback);
  
  // Return unsubscribe function
  return () => {
    const index = subscribers.indexOf(callback);
    if (index >= 0) {
      subscribers.splice(index, 1);
    }
  };
};

// Show toast notification for alert based on severity
export const showAlertToast = (alert: RiskAlert) => {
  let toastOptions: any = {
    duration: alert.severity === 'CRITICAL' ? Infinity : 
             alert.severity === 'HIGH' ? 8000 : 
             alert.severity === 'MEDIUM' ? 5000 : 3000,
  };
  
  // Different icon based on alert type/severity
  switch (alert.severity) {
    case 'CRITICAL':
      toastOptions.icon = '🚨';
      toast.error(
        `${alert.type}: ${alert.message}`, 
        toastOptions
      );
      break;
    case 'HIGH':
      toastOptions.icon = '⚠️';
      toast.error(
        `${alert.type}: ${alert.message}`, 
        toastOptions
      );
      break;
    case 'MEDIUM':
      toastOptions.icon = '⚠️';
      toast(
        `${alert.type}: ${alert.message}`, 
        toastOptions
      );
      break;
    default:
      toastOptions.icon = 'ℹ️';
      toast(
        `${alert.type}: ${alert.message}`, 
        toastOptions
      );
  }
};

// Handle incoming WebSocket alert
export const handleWebSocketAlert = (alertData: any) => {
  const alert: RiskAlert = {
    id: alertData.id || `alert-${Date.now()}`,
    type: alertData.type || RiskAlertType.MARKET_EVENT,
    message: alertData.message || 'Unknown alert',
    severity: alertData.severity || 'MEDIUM',
    timestamp: alertData.timestamp || new Date().toISOString(),
    source: alertData.source || 'system',
    details: alertData.details || {},
    acknowledged: false
  };
  
  addAlert(alert);
  showAlertToast(alert);
  
  return alert;
};
