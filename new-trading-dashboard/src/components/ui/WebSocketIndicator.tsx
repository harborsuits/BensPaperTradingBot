import React from 'react';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import Tooltip from './Tooltip';

interface WebSocketIndicatorProps {
  showText?: boolean;
  size?: number;
  className?: string;
}

const WebSocketIndicator: React.FC<WebSocketIndicatorProps> = ({ 
  showText = true,
  size = 16,
  className = ''
}) => {
  const { connectionStatus, reconnect } = useWebSocket();
  
  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="text-green-500" size={size} />;
      case 'connecting':
        return <Loader2 className="text-yellow-500 animate-spin" size={size} />;
      case 'disconnected':
      case 'error':
        return <WifiOff className="text-red-500" size={size} />;
      default:
        return <WifiOff className="text-gray-500" size={size} />;
    }
  };
  
  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'disconnected':
        return 'Disconnected';
      case 'error':
        return 'Connection Error';
      default:
        return 'Unknown Status';
    }
  };

  const getTooltipText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Receiving real-time updates';
      case 'connecting':
        return 'Establishing connection...';
      case 'disconnected':
        return 'Not receiving real-time updates - click to reconnect';
      case 'error':
        return 'Connection error - click to retry';
      default:
        return 'WebSocket connection status unknown';
    }
  };

  const handleReconnectClick = () => {
    if (connectionStatus !== 'connected' && connectionStatus !== 'connecting') {
      reconnect();
    }
  };

  return (
    <Tooltip content={getTooltipText()}>
      <div 
        className={`flex items-center cursor-pointer ${className}`}
        onClick={handleReconnectClick}
        role="button"
        aria-label={`WebSocket ${getStatusText()}`}
      >
        {showText && (
          <span className="text-sm mr-2">
            {getStatusText()}
          </span>
        )}
        {getStatusIcon()}
      </div>
    </Tooltip>
  );
};

export default WebSocketIndicator;
