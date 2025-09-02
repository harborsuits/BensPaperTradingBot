import React, { useState, useEffect } from 'react';
import { 
  useWebSocketStore, 
  useWebSocketChannel, 
  useWebSocketEvent, 
  useWebSocketStatus,
  WebSocketChannel
} from '../../services/websocketManager';
import { WebSocketEventType } from '../../services/websocketService';

// Status badge with appropriate colors
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  let bgColor = 'bg-gray-500';
  let textColor = 'text-white';
  
  if (status === 'connected') {
    bgColor = 'bg-green-500';
  } else if (status === 'disconnected') {
    bgColor = 'bg-red-500';
  } else if (status === 'reconnecting') {
    bgColor = 'bg-yellow-500';
    textColor = 'text-black';
  }
  
  return (
    <span className={`px-2 py-1 rounded-md text-xs font-semibold ${bgColor} ${textColor}`}>
      {status}
    </span>
  );
};

// Component to display recent messages
interface Message {
  id: string;
  type: string;
  timestamp: string;
  data: any;
}

const MessageList: React.FC<{ messages: Message[], title: string }> = ({ messages, title }) => {
  return (
    <div className="mt-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">{title}</h3>
      <div className="bg-gray-100 rounded-md p-3 h-40 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="text-gray-500 text-sm">No messages received yet</p>
        ) : (
          <ul className="space-y-2">
            {messages.map(msg => (
              <li key={msg.id} className="text-xs">
                <span className="font-medium">{msg.type}</span> • 
                <span className="text-gray-500"> {new Date(msg.timestamp).toLocaleTimeString()}</span>
                <pre className="mt-1 bg-gray-200 p-1 rounded text-xs overflow-x-auto">
                  {JSON.stringify(msg.data, null, 2)}
                </pre>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

// Channel subscription component
const ChannelSubscription: React.FC<{
  channel: WebSocketChannel;
  onToggle: () => void;
  isSubscribed: boolean;
}> = ({ channel, onToggle, isSubscribed }) => {
  return (
    <div className="flex items-center space-x-2">
      <input
        type="checkbox"
        id={`channel-${channel}`}
        checked={isSubscribed}
        onChange={onToggle}
        className="rounded text-blue-600"
      />
      <label htmlFor={`channel-${channel}`} className="text-sm">
        {channel}
      </label>
    </div>
  );
};

// Main WebSocket Monitor component
const WebSocketMonitor: React.FC = () => {
  const { status, connect, disconnect, isConnected } = useWebSocketStatus();
  const { subscribeToChannel, unsubscribeFromChannel } = useWebSocketStore();
  
  // Track subscribed channels
  const [subscribedChannels, setSubscribedChannels] = useState<Set<WebSocketChannel>>(new Set());
  
  // Messages received
  const [messages, setMessages] = useState<Message[]>([]);
  
  // Server URL
  const [serverUrl, setServerUrl] = useState('ws://localhost:8000/ws');
  
  // Toggle channel subscription
  const toggleChannel = (channel: WebSocketChannel) => {
    setSubscribedChannels(prev => {
      const newSet = new Set(prev);
      if (newSet.has(channel)) {
        newSet.delete(channel);
        unsubscribeFromChannel(channel);
      } else {
        newSet.add(channel);
        subscribeToChannel(channel);
      }
      return newSet;
    });
  };
  
  // Add a new message to our list
  const addMessage = (type: string, data: any) => {
    setMessages(prev => {
      const newMessages = [
        {
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          type,
          timestamp: new Date().toISOString(),
          data
        },
        ...prev
      ];
      
      // Keep only the last 50 messages
      return newMessages.slice(0, 50);
    });
  };
  
  // Listen for portfolio updates
  useWebSocketEvent('portfolio_update', (data) => {
    addMessage('portfolio_update', data);
  });
  
  // Listen for trade executions
  useWebSocketEvent('trade_executed', (data) => {
    addMessage('trade_executed', data);
  });
  
  // Listen for market context changes
  useWebSocketEvent('market_regime_change', (data) => {
    addMessage('market_regime_change', data);
  });
  
  // Listen for news alerts
  useWebSocketEvent('news_alert', (data) => {
    addMessage('news_alert', data);
  });
  
  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">WebSocket Monitor</h2>
      
      <div className="flex items-center mb-4 space-x-2">
        <StatusBadge status={status} />
        <span className="text-sm">Connection status</span>
      </div>
      
      <div className="flex flex-col space-y-4 md:flex-row md:space-y-0 md:space-x-4">
        <div className="flex-1">
          <div className="bg-white p-4 rounded-md shadow">
            <h3 className="text-md font-semibold mb-3">Connection</h3>
            
            <div className="mb-4">
              <label htmlFor="server-url" className="block text-sm font-medium text-gray-700 mb-1">
                Server URL
              </label>
              <input
                type="text"
                id="server-url"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md text-sm"
              />
            </div>
            
            <div className="flex space-x-2">
              <button
                onClick={() => connect()}
                disabled={isConnected}
                className={`px-3 py-1 rounded-md text-sm font-medium ${
                  isConnected 
                    ? 'bg-gray-200 text-gray-500 cursor-not-allowed' 
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                Connect
              </button>
              <button
                onClick={() => disconnect()}
                disabled={!isConnected}
                className={`px-3 py-1 rounded-md text-sm font-medium ${
                  !isConnected 
                    ? 'bg-gray-200 text-gray-500 cursor-not-allowed' 
                    : 'bg-red-600 text-white hover:bg-red-700'
                }`}
              >
                Disconnect
              </button>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-md shadow mt-4">
            <h3 className="text-md font-semibold mb-3">Channels</h3>
            <div className="grid grid-cols-2 gap-2">
              <ChannelSubscription 
                channel="market_data" 
                isSubscribed={subscribedChannels.has('market_data')}
                onToggle={() => toggleChannel('market_data')}
              />
              <ChannelSubscription 
                channel="trades" 
                isSubscribed={subscribedChannels.has('trades')}
                onToggle={() => toggleChannel('trades')}
              />
              <ChannelSubscription 
                channel="portfolio" 
                isSubscribed={subscribedChannels.has('portfolio')}
                onToggle={() => toggleChannel('portfolio')}
              />
              <ChannelSubscription 
                channel="strategies" 
                isSubscribed={subscribedChannels.has('strategies')}
                onToggle={() => toggleChannel('strategies')}
              />
              <ChannelSubscription 
                channel="market_context" 
                isSubscribed={subscribedChannels.has('market_context')}
                onToggle={() => toggleChannel('market_context')}
              />
              <ChannelSubscription 
                channel="alerts" 
                isSubscribed={subscribedChannels.has('alerts')}
                onToggle={() => toggleChannel('alerts')}
              />
              <ChannelSubscription 
                channel="logs" 
                isSubscribed={subscribedChannels.has('logs')}
                onToggle={() => toggleChannel('logs')}
              />
            </div>
          </div>
        </div>
        
        <div className="flex-1">
          <div className="bg-white p-4 rounded-md shadow">
            <h3 className="text-md font-semibold mb-3">Live Updates</h3>
            <MessageList messages={messages} title="Recent Messages" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebSocketMonitor;
