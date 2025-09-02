export type AlertSeverity = 'info' | 'warning' | 'error' | 'success';
export type AlertSource = 'system' | 'trade' | 'data' | 'security';
export type AlertStatus = 'new' | 'acknowledged' | 'resolved';

export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: AlertSeverity;
  source: AlertSource;
  status: AlertStatus;
  timestamp: string;
  metadata?: Record<string, any>;
  actionRequired?: boolean;
  actionUrl?: string;
  relatedSymbols?: string[];
}

export interface AlertsState {
  alerts: Alert[];
  unreadCount: number;
  hasErrors: boolean;
  lastChecked: string;
}

export interface AlertFilterOptions {
  severity?: AlertSeverity[];
  source?: AlertSource[];
  status?: AlertStatus[];
  timeRange?: {
    from: Date;
    to: Date;
  };
  search?: string;
}
