import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { ScrollArea } from '@/components/ui/ScrollArea';
import { Button } from '@/components/ui/Button';
import { 
  Clock, ArrowDown, 
  RefreshCw, AlertTriangle, Layers
} from 'lucide-react';
import { useWebSocketChannel, useWebSocketChannelMessage } from '@/services/websocketManager';

// Types for event system data
interface EventBusEntry {
  id: string;
  timestamp: string;
  event_type: string;
  source: string;
  destination?: string;
  payload_summary: string;
  status: 'pending' | 'processed' | 'error';
}

interface QueueStats {
  id: string;
  name: string;
  total_messages: number;
  processed_messages: number;
  pending_messages: number;
  errors: number;
  average_processing_time_ms: number;
}

interface ChannelStats {
  id: string;
  name: string;
  subscribers: number;
  messages_published: number;
  last_message_time?: string;
}

const EventSystemMonitor: React.FC = () => {
  // State variables
  const [eventBusEntries, setEventBusEntries] = useState<EventBusEntry[]>([]);
  const [queueStats, setQueueStats] = useState<QueueStats[]>([]);
  const [channelStats, setChannelStats] = useState<ChannelStats[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const [eventTypeFilter, setEventTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [isRecording, setIsRecording] = useState(true);
  const maxEvents = 100;

  // Subscribe to the events channel (using a type assertion as 'events' isn't in the standard channels)
  useWebSocketChannel('events' as any);

  // Fetch initial data
  useEffect(() => {
    const fetchEventSystemData = async () => {
      try {
        // Fetch event bus entries
        const entriesResponse = await fetch('/api/events/recent');
        const queuesResponse = await fetch('/api/events/queues');
        const channelsResponse = await fetch('/api/events/channels');
        
        if (entriesResponse.ok && queuesResponse.ok && channelsResponse.ok) {
          const entriesData = await entriesResponse.json();
          const queuesData = await queuesResponse.json();
          const channelsData = await channelsResponse.json();
          
          setEventBusEntries(entriesData);
          setQueueStats(queuesData);
          setChannelStats(channelsData);
          setLastUpdateTime(new Date());
        }
      } catch (error) {
        console.error('Error fetching event system data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchEventSystemData();
  }, []);

  // Listen for event system updates via WebSocket
  useWebSocketChannelMessage('events' as any, (message) => {
    if (!isRecording) return;
    
    // Type assertion to work with the message payload
    const typedMessage = message as any;
    
    if (typedMessage.event === 'event_bus_entry') {
      // Add a new event bus entry
      const newEntry = typedMessage.payload?.event;
      if (newEntry) {
        setEventBusEntries(prevEntries => {
          const updated = [newEntry, ...prevEntries].slice(0, maxEvents);
          return updated;
        });
        setLastUpdateTime(new Date());
      }
    } else if (typedMessage.event === 'queue_stats_update') {
      // Update queue statistics
      const queues = typedMessage.payload?.queues;
      if (queues) {
        setQueueStats(queues);
        setLastUpdateTime(new Date());
      }
    } else if (typedMessage.event === 'channel_stats_update') {
      // Update channel statistics
      const channels = typedMessage.payload?.channels;
      if (channels) {
        setChannelStats(channels);
        setLastUpdateTime(new Date());
      }
    } else if (typedMessage.event === 'event_status_change') {
      // Update an existing event entry status
      const event_id = typedMessage.payload?.event_id;
      const new_status = typedMessage.payload?.new_status;
      if (event_id && new_status) {
        setEventBusEntries(prevEntries => 
          prevEntries.map(entry => 
            entry.id === event_id 
              ? { ...entry, status: new_status } 
              : entry
          )
        );
      }
      setLastUpdateTime(new Date());
    }
  });

  // Filter event bus entries
  const getFilteredEvents = () => {
    return eventBusEntries.filter(entry => {
      const matchesType = eventTypeFilter === 'all' || entry.event_type === eventTypeFilter;
      const matchesStatus = statusFilter === 'all' || entry.status === statusFilter;
      return matchesType && matchesStatus;
    });
  };

  // Get unique event types for filter dropdown
  const getEventTypes = () => {
    const types = new Set<string>();
    eventBusEntries.forEach(entry => types.add(entry.event_type));
    return Array.from(types);
  };

  // Get status badge
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'processed':
        return (
          <Badge variant="outline" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
            Processed
          </Badge>
        );
      case 'pending':
        return (
          <Badge variant="outline" className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
            Pending
          </Badge>
        );
      case 'error':
        return (
          <Badge variant="outline" className="bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300">
            Error
          </Badge>
        );
      default:
        return null;
    }
  };

  // Clear events
  const handleClearEvents = () => {
    setEventBusEntries([]);
  };

  // Toggle recording
  const handleRecordingToggle = () => {
    setIsRecording(!isRecording);
  };

  // Refresh data
  const handleRefresh = async () => {
    setIsLoading(true);
    
    try {
      const entriesResponse = await fetch('/api/events/recent');
      const queuesResponse = await fetch('/api/events/queues');
      const channelsResponse = await fetch('/api/events/channels');
      
      if (entriesResponse.ok && queuesResponse.ok && channelsResponse.ok) {
        const entriesData = await entriesResponse.json();
        const queuesData = await queuesResponse.json();
        const channelsData = await channelsResponse.json();
        
        setEventBusEntries(entriesData);
        setQueueStats(queuesData);
        setChannelStats(channelsData);
        setLastUpdateTime(new Date());
      }
    } catch (error) {
      console.error('Error refreshing event system data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center text-lg">
            <Layers size={18} className="mr-2" />
            Event System Monitor
          </CardTitle>
          <div className="flex items-center space-x-2">
            {lastUpdateTime && (
              <span className="text-xs text-muted-foreground flex items-center">
                <Clock size={12} className="mr-1" />
                Last: {lastUpdateTime.toLocaleTimeString()}
              </span>
            )}
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleRefresh}
              className="h-7 px-2"
            >
              <RefreshCw size={14} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="events">
          <TabsList className="mb-2">
            <TabsTrigger value="events">Events</TabsTrigger>
            <TabsTrigger value="queues">Message Queues</TabsTrigger>
            <TabsTrigger value="channels">Channels</TabsTrigger>
          </TabsList>
          
          <TabsContent value="events" className="space-y-4">
            {/* Controls */}
            <div className="flex flex-wrap gap-2 items-center justify-between">
              <div className="flex flex-wrap gap-2 items-center">
                <select 
                  value={eventTypeFilter} 
                  onChange={(e) => setEventTypeFilter(e.target.value)}
                  className="text-xs border rounded px-2 py-1 bg-background"
                  aria-label="Filter by event type"
                >
                  <option value="all">All Event Types</option>
                  {getEventTypes().map(type => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
                
                <select 
                  value={statusFilter} 
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="text-xs border rounded px-2 py-1 bg-background"
                  aria-label="Filter by status"
                >
                  <option value="all">All Statuses</option>
                  <option value="pending">Pending</option>
                  <option value="processed">Processed</option>
                  <option value="error">Error</option>
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
                onClick={handleClearEvents}
              >
                Clear
              </Button>
            </div>
            
            {/* Events */}
            <ScrollArea className="h-[400px] border rounded-lg p-2 bg-muted/10">
              {isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="h-16 bg-muted/20 animate-pulse rounded-md"></div>
                  ))}
                </div>
              ) : getFilteredEvents().length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <ArrowDown size={24} className="mb-2" />
                  <p>No events matching filter</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {getFilteredEvents().map((entry) => (
                    <div 
                      key={entry.id} 
                      className="p-2 rounded-md text-xs border bg-card/50"
                    >
                      <div className="flex justify-between items-start mb-1">
                        <div className="flex items-center gap-1">
                          <Badge variant="secondary">
                            {entry.event_type}
                          </Badge>
                          {getStatusBadge(entry.status)}
                        </div>
                        <span className="text-muted-foreground text-xs">
                          {new Date(entry.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-x-2 gap-y-1 mt-2 text-xs">
                        <span className="text-muted-foreground">Source:</span>
                        <span>{entry.source}</span>
                        
                        {entry.destination && (
                          <>
                            <span className="text-muted-foreground">Destination:</span>
                            <span>{entry.destination}</span>
                          </>
                        )}
                        
                        <span className="text-muted-foreground">Payload:</span>
                        <span className="truncate">{entry.payload_summary}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="queues" className="space-y-4">
            {/* Message Queues */}
            <ScrollArea className="h-[400px] border rounded-lg p-2 bg-muted/10">
              {isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="h-24 bg-muted/20 animate-pulse rounded-md"></div>
                  ))}
                </div>
              ) : queueStats.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <AlertTriangle size={24} className="mb-2" />
                  <p>No message queues available</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {queueStats.map((queue) => (
                    <div 
                      key={queue.id} 
                      className="p-3 border rounded-md"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-medium">{queue.name}</h3>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="flex justify-between items-center p-2 bg-muted/10 rounded">
                          <span className="text-muted-foreground">Total Messages:</span>
                          <Badge variant="secondary">{queue.total_messages}</Badge>
                        </div>
                        
                        <div className="flex justify-between items-center p-2 bg-muted/10 rounded">
                          <span className="text-muted-foreground">Processed:</span>
                          <Badge 
                            variant="outline" 
                            className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                          >
                            {queue.processed_messages}
                          </Badge>
                        </div>
                        
                        <div className="flex justify-between items-center p-2 bg-muted/10 rounded">
                          <span className="text-muted-foreground">Pending:</span>
                          <Badge 
                            variant="outline" 
                            className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                          >
                            {queue.pending_messages}
                          </Badge>
                        </div>
                        
                        <div className="flex justify-between items-center p-2 bg-muted/10 rounded">
                          <span className="text-muted-foreground">Errors:</span>
                          <Badge 
                            variant="outline" 
                            className={queue.errors > 0 
                              ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300" 
                              : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                            }
                          >
                            {queue.errors}
                          </Badge>
                        </div>
                        
                        <div className="col-span-2 flex justify-between items-center p-2 bg-muted/10 rounded">
                          <span className="text-muted-foreground">Avg Processing Time:</span>
                          <span className="font-medium">{queue.average_processing_time_ms.toFixed(2)}ms</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="channels" className="space-y-4">
            {/* Channels */}
            <ScrollArea className="h-[400px] border rounded-lg p-2 bg-muted/10">
              {isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="h-16 bg-muted/20 animate-pulse rounded-md"></div>
                  ))}
                </div>
              ) : channelStats.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <AlertTriangle size={24} className="mb-2" />
                  <p>No channels available</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-3">
                  {channelStats.map((channel) => (
                    <div 
                      key={channel.id} 
                      className="p-3 border rounded-md"
                    >
                      <div className="flex justify-between items-center">
                        <h3 className="font-medium">{channel.name}</h3>
                        <Badge variant="secondary">
                          {channel.subscribers} subscribers
                        </Badge>
                      </div>
                      
                      <div className="mt-2 flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Messages published:</span>
                        <span className="font-medium">{channel.messages_published}</span>
                      </div>
                      
                      {channel.last_message_time && (
                        <div className="mt-1 flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Last message:</span>
                          <span className="text-xs">
                            {new Date(channel.last_message_time).toLocaleTimeString()}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default EventSystemMonitor;
