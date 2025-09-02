import { useEffect } from 'react';
import { 
  useWebSocketMessage, 
  useWebSocketChannel 
} from '@/services/websocket';
import { 
  Strategy, 
  TradeCandidate, 
  LogEvent, 
  AlertNotification,
  MarketContext,
  PriceUpdateModel
} from '@/types/api.types';

/**
 * Custom hook to subscribe to strategy updates via WebSocket
 */
export const useStrategyUpdates = (
  onStrategyUpdate: (strategy: Strategy) => void
) => {
  // Subscribe to the strategies channel
  const { isConnected } = useWebSocketChannel('strategies');

  // Add message handler for strategy updates
  useWebSocketMessage<Strategy>(
    'strategy_update',
    (message) => {
      onStrategyUpdate(message.data);
    },
    [onStrategyUpdate]
  );

  return { isConnected };
};

/**
 * Custom hook to subscribe to trade decision updates via WebSocket
 */
export const useTradeDecisionUpdates = (
  onDecisionUpdate: (decision: any) => void
) => {
  // Subscribe to the decisions channel
  const { isConnected } = useWebSocketChannel('decisions');

  // Add message handler for decision updates
  useWebSocketMessage<any>(
    'decision_update',
    (message) => {
      onDecisionUpdate(message.data);
    },
    [onDecisionUpdate]
  );

  return { isConnected };
};

/**
 * Custom hook to subscribe to log events via WebSocket
 */
export const useLogUpdates = (
  onLogEvent: (log: LogEvent) => void
) => {
  // Subscribe to the logs channel
  const { isConnected } = useWebSocketChannel('logs');

  // Add message handler for log events
  useWebSocketMessage<LogEvent>(
    'log_event',
    (message) => {
      onLogEvent(message.data);
    },
    [onLogEvent]
  );

  return { isConnected };
};

/**
 * Custom hook to subscribe to alert notifications via WebSocket
 */
export const useAlertUpdates = (
  onNewAlert: (alert: AlertNotification) => void,
  onAlertUpdate: (alert: AlertNotification) => void
) => {
  // Subscribe to the alerts channel
  const { isConnected } = useWebSocketChannel('alerts');

  // Add message handler for new alerts
  useWebSocketMessage<AlertNotification>(
    'new_alert',
    (message) => {
      onNewAlert(message.data);
    },
    [onNewAlert]
  );

  // Add message handler for alert updates (e.g., acknowledged)
  useWebSocketMessage<AlertNotification>(
    'alert_update',
    (message) => {
      onAlertUpdate(message.data);
    },
    [onAlertUpdate]
  );

  return { isConnected };
};

/**
 * Custom hook to subscribe to context updates via WebSocket
 */
export const useContextUpdates = (
  onRegimeChange?: (regime: any) => void,
  onFeatureUpdate?: (features: any) => void,
  onNewsUpdate?: (news: any) => void,
  onAnomalyDetected?: (anomaly: any) => void
) => {
  // Subscribe to the context channel
  const { isConnected } = useWebSocketChannel('context');

  // Add message handlers for different context updates
  useWebSocketMessage<any>(
    'regime_change',
    (message) => {
      if (onRegimeChange) onRegimeChange(message.data);
    },
    [onRegimeChange]
  );

  useWebSocketMessage<any>(
    'feature_update',
    (message) => {
      if (onFeatureUpdate) onFeatureUpdate(message.data);
    },
    [onFeatureUpdate]
  );

  useWebSocketMessage<any>(
    'news_update',
    (message) => {
      if (onNewsUpdate) onNewsUpdate(message.data);
    },
    [onNewsUpdate]
  );

  useWebSocketMessage<any>(
    'anomaly_detected',
    (message) => {
      if (onAnomalyDetected) onAnomalyDetected(message.data);
    },
    [onAnomalyDetected]
  );

  return { isConnected };
};

/**
 * Custom hook to subscribe to data updates via WebSocket
 */
export const useDataUpdates = (
  onPriceUpdate?: (price: PriceUpdateModel) => void,
  onSourceStatusUpdate?: (status: any) => void,
  onMetricsUpdate?: (metrics: any) => void
) => {
  // Subscribe to the data channel
  const { isConnected } = useWebSocketChannel('data');

  // Add message handlers for different data updates
  useWebSocketMessage<PriceUpdateModel>(
    'price_update',
    (message) => {
      if (onPriceUpdate) onPriceUpdate(message.data);
    },
    [onPriceUpdate]
  );

  useWebSocketMessage<any>(
    'ingestion_sources_status',
    (message) => {
      if (onSourceStatusUpdate) onSourceStatusUpdate(message.data);
    },
    [onSourceStatusUpdate]
  );

  useWebSocketMessage<any>(
    'ingestion_metrics',
    (message) => {
      if (onMetricsUpdate) onMetricsUpdate(message.data);
    },
    [onMetricsUpdate]
  );

  return { isConnected };
};
