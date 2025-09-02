import React, { useEffect } from 'react';
// No Helmet component needed here
import BrokerPerformance from '@/components/broker/BrokerPerformance';
import RiskAlertsPanel from '@/components/broker/RiskAlertsPanel';
import StrategyMonitor from '@/components/strategy/StrategyMonitor';
import { registerWebSocketEventHandlers } from '@/services/websocketEvents';
import { useWebSocket } from '@/contexts/WebSocketContext';

const BrokerPage: React.FC = () => {
  // Get the WebSocket manager from context
  const websocketContext = useWebSocket();
  const webSocketManager = websocketContext?.manager;
  
  useEffect(() => {
    if (!webSocketManager) return;
    
    // Register WebSocket handlers for real-time updates
    registerWebSocketEventHandlers(webSocketManager);
    
    // Subscribe to broker and strategy-specific channels
    webSocketManager.subscribe('broker');
    webSocketManager.subscribe('strategy');
    webSocketManager.subscribe('alert');
    
    return () => {
      if (!webSocketManager) return;
      // Unsubscribe from channels when component unmounts
      webSocketManager.unsubscribe('broker');
      webSocketManager.unsubscribe('strategy');
      webSocketManager.unsubscribe('alert');
    };
  }, [webSocketManager]);
  
  return (
    <div className="space-y-8 pb-10">
          {/* Page title set in parent layout */}
      
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Broker & Strategy Management</h1>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BrokerPerformance className="lg:col-span-1" />
        <RiskAlertsPanel className="lg:col-span-1" />
      </div>
      
      <StrategyMonitor className="w-full" />
    </div>
  );
};

export default BrokerPage;
