import { WebSocketManager } from './websocketManager';
// Use the global instance from the app
const queryClient = window.__APP_QUERY_CLIENT__;
import { toast } from 'react-hot-toast';
import { handleWebSocketAlert } from './riskAlerts';

// Define WebSocket event types
export type WebSocketEventType = 
  | 'portfolio_update'
  | 'trade_executed'
  | 'market_regime_change'
  | 'sentiment_update'
  | 'market_features_update'
  | 'broker_performance_update'
  | 'broker_status_change'
  | 'strategy_update'
  | 'strategy_signal'
  | 'risk_alert';

// Register event handlers for WebSocket events
export const registerWebSocketEventHandlers = (wsManager: WebSocketManager) => {
  // Portfolio updates
  wsManager.registerEventHandler('portfolio_update', (data) => {
    queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    console.log('Portfolio updated via WebSocket', data);
  });
  
  // Trade executed
  wsManager.registerEventHandler('trade_executed', (data) => {
    queryClient.invalidateQueries({ queryKey: ['trades'] });
    queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    
    // Show toast notification for trade
    toast.success(`Trade Executed: ${data.direction === 'BUY' ? 'Bought' : 'Sold'} ${data.quantity} ${data.symbol} @ $${data.price.toFixed(2)}`);
    
    console.log('Trade executed via WebSocket', data);
  });
  
  // Market regime change
  wsManager.registerEventHandler('market_regime_change', (data) => {
    queryClient.invalidateQueries({ queryKey: ['market-regime'] });
    
    // Show toast notification for significant regime changes
    if (data.confidence > 0.7) {
      toast(`Market Regime Change: New regime: ${data.regime} (${Math.round(data.confidence * 100)}% confidence)`);
    }
    
    console.log('Market regime updated via WebSocket', data);
  });
  
  // Sentiment update
  wsManager.registerEventHandler('sentiment_update', (data) => {
    queryClient.invalidateQueries({ queryKey: ['sentiment'] });
    console.log('Sentiment updated via WebSocket', data);
  });
  
  // Market features update
  wsManager.registerEventHandler('market_features_update', (data) => {
    queryClient.invalidateQueries({ queryKey: ['market-features'] });
    console.log('Market features updated via WebSocket', data);
  });
  
  // Broker performance update
  wsManager.registerEventHandler('broker_performance_update', (data) => {
    queryClient.invalidateQueries({ queryKey: ['broker-performance'] });
    console.log('Broker performance updated via WebSocket', data);
  });
  
  // Broker status change
  wsManager.registerEventHandler('broker_status_change', (data) => {
    queryClient.invalidateQueries({ queryKey: ['broker-performance'] });
    
    // Show toast notification for status changes
    const statusMap = {
      'online': 'Online',
      'offline': 'Offline',
      'degraded': 'Degraded Performance',
      'circuit_broken': 'Circuit Breaker Triggered'
    };
    
    const statusText = statusMap[data.status] || data.status;
    
    if (data.status === 'offline' || data.status === 'circuit_broken') {
      toast.error(`Broker Status Change: ${data.broker}: ${statusText}`);
    } else if (data.status === 'degraded') {
      toast(`Broker Status Change: ${data.broker}: ${statusText}`, { icon: '⚠️' });
    } else {
      toast.success(`Broker Status Change: ${data.broker}: ${statusText}`);
    }
    
    console.log('Broker status changed via WebSocket', data);
  });
  
  // Strategy update
  wsManager.registerEventHandler('strategy_update', (data) => {
    queryClient.invalidateQueries({ queryKey: ['active-strategies'] });
    queryClient.invalidateQueries({ queryKey: ['strategies'] });
    
    console.log('Strategy updated via WebSocket', data);
  });
  
  // Strategy signal
  wsManager.registerEventHandler('strategy_signal', (data) => {
    queryClient.invalidateQueries({ queryKey: ['active-strategies'] });
    
    // Only show notifications for strong signals
    if (data.strength > 0.7) {
      const directionDisplay = data.direction.charAt(0).toUpperCase() + data.direction.slice(1);
      const icon = data.direction === 'long' ? '📈' : 
                  data.direction === 'short' ? '📉' : 'ℹ️';
      
      toast(`${data.strategy}: New Signal - ${directionDisplay} (${Math.round(data.strength * 100)}% strength)`, { icon });
    }
    
    console.log('Strategy signal received via WebSocket', data);
  });
  
  // Risk alert
  wsManager.registerEventHandler('risk_alert', (data) => {
    // Use the risk alert handler to process and display the alert
    handleWebSocketAlert(data);
    
    console.log('Risk alert received via WebSocket', data);
  });
};
