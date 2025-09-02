import React, { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Activity, Wifi, WifiOff, X, Clock, ArrowDown } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { ScrollArea } from '@/components/ui/ScrollArea';
import { useWebSocketStore, useWebSocketChannel, useWebSocketChannelMessage, useWebSocketStatus } from '@/services/websocketManager';

interface WebSocketMessage {
  id: string;
  timestamp: Date;
  direction: 'in' | 'out';
  channel: string;
  event: string;
  payload: any;
}

const WebSocketMonitor: React.FC = () => {
  const { connect, disconnect, isConnected } = useWebSocketStatus();
  const { subscribeToChannel, unsubscribeFromChannel } = useWebSocketStore();
  
  
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [filteredMessages, setFilteredMessages] = useState<WebSocketMessage[]>([]);
  const [channelFilter, setChannelFilter] = useState<string>('all');
  const [directionFilter, setDirectionFilter] = useState<'all' | 'in' | 'out'>('all');
  const [isRecording, setIsRecording] = useState(true);
  const [lastActivity, setLastActivity] = useState<Date | null>(null);
  const [subscribedChannels, setSubscribedChannels] = useState<string[]>([]);
  const [newChannelName, setNewChannelName] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const maxMessages = 100;

  // Listen for WebSocket messages
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (!isRecording) return;
      
      try {
        const data = JSON.parse(event.data);
        const newMessage: WebSocketMessage = {
          id: crypto.randomUUID(),
          timestamp: new Date(),
          direction: 'in',
          channel: data.channel || 'unknown',
          event: data.event || 'unknown',
          payload: data.payload || data
        };
        
        setMessages(prev => {
          const updated = [newMessage, ...prev].slice(0, maxMessages);
          return updated;
        });
        setLastActivity(new Date());
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    // Update subscribed channels list
    const updateChannels = () => {
      // Get current subscribed channels from localStorage or another method
      try {
        const channels = localStorage.getItem('websocket_channels');
        if (channels) {
          setSubscribedChannels(JSON.parse(channels));
        }
      } catch (error) {
        console.error('Failed to get subscribed channels:', error);
      }
    };

    // Add event listeners
    window.addEventListener('websocket-message', handleMessage as EventListener);
    // Setup event listeners for custom events
    const onSubscribed = () => updateChannels();
    const onUnsubscribed = () => updateChannels();
    const onConnected = () => updateChannels();
    
    window.addEventListener('websocket-subscribed', onSubscribed);
    window.addEventListener('websocket-unsubscribed', onUnsubscribed);
    window.addEventListener('websocket-connected', onConnected);

    // Initial channels
    updateChannels();

    return () => {
      window.removeEventListener('websocket-message', handleMessage as EventListener);
      window.removeEventListener('websocket-subscribed', onSubscribed);
      window.removeEventListener('websocket-unsubscribed', onUnsubscribed);
      window.removeEventListener('websocket-connected', onConnected);
    };
  }, [isRecording]);

  // Apply filters
  useEffect(() => {
    let filtered = [...messages];
    
    if (channelFilter !== 'all') {
      filtered = filtered.filter(msg => msg.channel === channelFilter);
    }
    
    if (directionFilter !== 'all') {
      filtered = filtered.filter(msg => msg.direction === directionFilter);
    }
    
    setFilteredMessages(filtered);
  }, [messages, channelFilter, directionFilter]);

  // Handle connection toggle
  const handleConnectionToggle = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };
  
  // Get subscribed channels - would normally come from the WebSocket manager
  const getSubscribedChannels = () => {
    try {
      const channels = localStorage.getItem('websocket_channels');
      return channels ? JSON.parse(channels) : [];
    } catch (error) {
      console.error('Failed to get subscribed channels:', error);
      return [];
    }
  };

  // Clear messages
  const handleClearMessages = () => {
    setMessages([]);
    setFilteredMessages([]);
  };

  // Toggle recording
  const handleRecordingToggle = () => {
    setIsRecording(!isRecording);
  };

  // Subscribe to channel
  const handleSubscribe = () => {
    if (newChannelName.trim()) {
      subscribeToChannel(newChannelName.trim() as any);
      
      // Update local storage for this example
      try {
        const channels = getSubscribedChannels();
        if (!channels.includes(newChannelName.trim())) {
          channels.push(newChannelName.trim());
          localStorage.setItem('websocket_channels', JSON.stringify(channels));
          setSubscribedChannels(channels);
        }
      } catch (error) {
        console.error('Failed to save channel subscription:', error);
      }
      
      setNewChannelName('');
    }
  };

  // Unsubscribe from channel
  const handleUnsubscribe = (channel: string) => {
    unsubscribeFromChannel(channel as any);
    
    // Update local storage for this example
    try {
      const channels = getSubscribedChannels();
      const updatedChannels = channels.filter((c: string) => c !== channel);
      localStorage.setItem('websocket_channels', JSON.stringify(updatedChannels));
      setSubscribedChannels(updatedChannels);
    } catch (error) {
      console.error('Failed to remove channel subscription:', error);
    }
  };

  // Get status color
  const getStatusColor = () => {
    return isConnected ? 'text-green-500' : 'text-red-500';
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center text-lg">
            <Activity size={18} className="mr-2" />
            WebSocket Monitor
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge 
              variant="outline" 
              className={`${isConnected 
                ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' 
                : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'}`}
            >
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
            {lastActivity && (
              <span className="text-xs text-muted-foreground flex items-center">
                <Clock size={12} className="mr-1" />
                Last: {lastActivity.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="messages">
          <TabsList className="mb-2">
            <TabsTrigger value="messages">Messages</TabsTrigger>
            <TabsTrigger value="connections">Connections</TabsTrigger>
          </TabsList>
          
          <TabsContent value="messages" className="space-y-4">
            {/* Controls */}
            <div className="flex flex-wrap gap-2 items-center justify-between">
              <div className="flex flex-wrap gap-2 items-center">
                <select 
                  value={channelFilter} 
                  onChange={(e) => setChannelFilter(e.target.value)}
                  className="text-xs border rounded px-2 py-1 bg-background"
                  aria-label="Filter by channel"
                >
                  <option value="all">All Channels</option>
                  {subscribedChannels.map(channel => (
                    <option key={channel} value={channel}>
                      {channel}
                    </option>
                  ))}
                </select>
                
                <select 
                  value={directionFilter} 
                  onChange={(e) => setDirectionFilter(e.target.value as 'all' | 'in' | 'out')}
                  className="text-xs border rounded px-2 py-1 bg-background"
                  aria-label="Filter by direction"
                >
                  <option value="all">All Directions</option>
                  <option value="in">Incoming</option>
                  <option value="out">Outgoing</option>
                </select>
                
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleRecordingToggle}
                  className={isRecording ? 'bg-red-100 dark:bg-red-900/20' : ''}
                >
                  {isRecording ? 'Stop Recording' : 'Start Recording'}
                </Button>
              </div>
              
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleClearMessages}
              >
                Clear
              </Button>
            </div>
            
            {/* Messages */}
            <ScrollArea className="h-[400px] border rounded-lg p-2 bg-muted/10">
              {filteredMessages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <ArrowDown size={24} className="mb-2" />
                  <p>No messages yet</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {filteredMessages.map((message) => (
                    <div 
                      key={message.id} 
                      className={`p-2 rounded-md text-xs font-mono ${
                        message.direction === 'in' 
                          ? 'bg-blue-50 dark:bg-blue-950/30 border-l-2 border-blue-400'
                          : 'bg-green-50 dark:bg-green-950/30 border-l-2 border-green-400'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <div>
                          <Badge variant="outline" className="mr-1">
                            {message.channel}
                          </Badge>
                          <Badge variant="outline">
                            {message.event}
                          </Badge>
                        </div>
                        <span className="text-muted-foreground">
                          {message.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="bg-black/10 p-1 rounded overflow-x-auto">
                        <pre className="whitespace-pre-wrap break-words">
                          {JSON.stringify(message.payload, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div ref={messagesEndRef} />
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="connections" className="space-y-4">
            {/* Connection controls */}
            <div className="flex items-center gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleConnectionToggle}
                className={isConnected ? 'bg-red-100 dark:bg-red-900/20' : 'bg-green-100 dark:bg-green-900/20'}
              >
                {isConnected ? (
                  <>
                    <WifiOff size={16} className="mr-1" />
                    Disconnect
                  </>
                ) : (
                  <>
                    <Wifi size={16} className="mr-1" />
                    Connect
                  </>
                )}
              </Button>
            </div>
            
            {/* Channel subscription */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Channel Subscriptions</h3>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newChannelName}
                  onChange={(e) => setNewChannelName(e.target.value)}
                  placeholder="Channel name"
                  className="text-xs border rounded px-2 py-1 flex-1 bg-background"
                />
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleSubscribe}
                  disabled={!newChannelName.trim() || !isConnected}
                >
                  Subscribe
                </Button>
              </div>
              
              {/* Subscribed channels */}
              <div className="space-y-1 mt-2">
                {subscribedChannels.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No channels subscribed</p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {subscribedChannels.map(channel => (
                      <Badge 
                        key={channel} 
                        variant="secondary"
                        className="flex items-center gap-1"
                      >
                        {channel}
                        <button 
                          onClick={() => handleUnsubscribe(channel)}
                          className="ml-1 hover:text-red-500"
                          aria-label={`Unsubscribe from ${channel}`}
                        >
                          <X size={12} />
                        </button>
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Connection details */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Connection Details</h3>
              <div className="text-xs border rounded p-2 bg-muted/10">
                <div className="grid grid-cols-2 gap-1">
                  <span className="text-muted-foreground">Status:</span>
                  <span className={getStatusColor()}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                  
                  <span className="text-muted-foreground">URL:</span>
                  <span>ws://localhost:8000/ws</span>
                  
                  <span className="text-muted-foreground">Subscriptions:</span>
                  <span>{subscribedChannels.length}</span>
                  
                  {lastActivity && (
                    <>
                      <span className="text-muted-foreground">Last Activity:</span>
                      <span>{lastActivity.toLocaleTimeString()}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default WebSocketMonitor;
