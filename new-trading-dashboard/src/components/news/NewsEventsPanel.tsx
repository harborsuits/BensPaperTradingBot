import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Switch } from '@/components/ui/Switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { ScrollArea } from '@/components/ui/ScrollArea';
import { Separator } from '@/components/ui/Separator';
import { Alert, AlertDescription } from '@/components/ui/Alert';
import { Progress } from '@/components/ui/Progress';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Info,
  Settings,
  RefreshCw,
  Eye,
  EyeOff,
  Zap,
  Shield,
  Clock
} from 'lucide-react';

interface NewsEvent {
  id: string;
  title: string;
  content: string;
  source: string;
  timestamp: string;
  eventType: string;
  confidence: number;
  sentiment: 'positive' | 'negative' | 'neutral';
  impact: 'high' | 'medium' | 'low';
  symbols: string[];
  classification: {
    category: string;
    subcategory?: string;
    confidence: number;
  };
  reaction?: {
    expectedReturn: number;
    confidence: number;
    timeHorizon: string;
  };
}

interface NewsSystemStatus {
  enabled: boolean;
  eventClassifier: {
    status: 'active' | 'degraded' | 'offline';
    eventsProcessed: number;
    avgLatency: number;
  };
  reactionStats: {
    status: 'active' | 'updating' | 'offline';
    eventsTracked: number;
    lastUpdate: string;
  };
  newsNudge: {
    status: 'active' | 'circuit_breaker' | 'offline';
    activeNudges: number;
    avgNudgeStrength: number;
  };
  circuitBreaker: {
    triggered: boolean;
    reason?: string;
    autoResetTime?: string;
  };
}

export const NewsEventsPanel: React.FC = () => {
  const [events, setEvents] = useState<NewsEvent[]>([]);
  const [systemStatus, setSystemStatus] = useState<NewsSystemStatus | null>(null);
  const [isEnabled, setIsEnabled] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<NewsEvent | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch news events
  useEffect(() => {
    fetchEvents();
    fetchSystemStatus();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      fetchEvents();
      fetchSystemStatus();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchEvents = async () => {
    try {
      const response = await fetch('/api/news/events');
      if (response.ok) {
        const data = await response.json();
        setEvents(data.events || []);
      }
    } catch (error) {
      console.error('Failed to fetch news events:', error);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/news/status');
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
        setIsEnabled(data.enabled);
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleNewsSystem = async (enabled: boolean) => {
    try {
      const response = await fetch('/api/news/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });

      if (response.ok) {
        setIsEnabled(enabled);
        await fetchSystemStatus(); // Refresh status
      }
    } catch (error) {
      console.error('Failed to toggle news system:', error);
    }
  };

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'earnings': return <TrendingUp className="w-4 h-4" />;
      case 'guidance': return <TrendingDown className="w-4 h-4" />;
      case 'regulatory': return <Shield className="w-4 h-4" />;
      case 'product': return <Zap className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-50 border-green-200';
      case 'negative': return 'text-red-600 bg-red-50 border-red-200';
      case 'neutral': return 'text-gray-600 bg-gray-50 border-gray-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return <Badge variant="default" className="bg-green-100 text-green-800">High</Badge>;
    if (confidence >= 0.6) return <Badge variant="secondary">Medium</Badge>;
    return <Badge variant="outline">Low</Badge>;
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* System Status Header */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold">News Intelligence System</h3>
            <Switch
              checked={isEnabled}
              onCheckedChange={toggleNewsSystem}
            />
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              {showDetails ? 'Hide' : 'Details'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchEvents}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* System Status Indicators */}
        {systemStatus && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                systemStatus.eventClassifier.status === 'active' ? 'bg-green-500' :
                systemStatus.eventClassifier.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm">Event Classifier</span>
              <Badge variant="outline" className="text-xs">
                {systemStatus.eventClassifier.eventsProcessed} events
              </Badge>
            </div>

            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                systemStatus.reactionStats.status === 'active' ? 'bg-green-500' :
                systemStatus.reactionStats.status === 'updating' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm">Reaction Stats</span>
              <Badge variant="outline" className="text-xs">
                {systemStatus.reactionStats.eventsTracked} tracked
              </Badge>
            </div>

            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                systemStatus.newsNudge.status === 'active' ? 'bg-green-500' :
                systemStatus.newsNudge.status === 'circuit_breaker' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm">News Nudge</span>
              <Badge variant="outline" className="text-xs">
                {systemStatus.newsNudge.activeNudges} active
              </Badge>
            </div>
          </div>
        )}

        {/* Circuit Breaker Alert */}
        {systemStatus?.circuitBreaker.triggered && (
          <Alert className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              News system circuit breaker triggered: {systemStatus.circuitBreaker.reason}
              {systemStatus.circuitBreaker.autoResetTime &&
                ` - Auto-reset at ${new Date(systemStatus.circuitBreaker.autoResetTime).toLocaleTimeString()}`}
            </AlertDescription>
          </Alert>
        )}
      </Card>

      {/* News Events Display */}
      <Card className="p-4">
        <Tabs defaultValue="recent" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="recent">Recent Events</TabsTrigger>
            <TabsTrigger value="high-impact">High Impact</TabsTrigger>
            <TabsTrigger value="watchlist">Watchlist</TabsTrigger>
          </TabsList>

          <TabsContent value="recent" className="mt-4">
            <ScrollArea className="h-96">
              <div className="space-y-3">
                {events.slice(0, 10).map((event) => (
                  <div
                    key={event.id}
                    className="p-3 border rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => setSelectedEvent(event)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getEventIcon(event.eventType)}
                        <span className="font-medium text-sm">{event.title}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {getConfidenceBadge(event.confidence)}
                        <Badge
                          variant="outline"
                          className={getSentimentColor(event.sentiment)}
                        >
                          {event.sentiment}
                        </Badge>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-xs text-gray-600">
                      <span>{event.source} • {new Date(event.timestamp).toLocaleString()}</span>
                      <div className="flex items-center gap-2">
                        <span>Symbols: {event.symbols.join(', ')}</span>
                        <Badge variant="outline" className="text-xs">
                          {event.classification.category}
                        </Badge>
                      </div>
                    </div>

                    {event.reaction && (
                      <div className="mt-2 flex items-center gap-2 text-xs">
                        <Clock className="w-3 h-3" />
                        <span>Expected: {event.reaction.expectedReturn > 0 ? '+' : ''}{(event.reaction.expectedReturn * 100).toFixed(1)}%</span>
                        <span>• {event.reaction.timeHorizon}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="high-impact" className="mt-4">
            <ScrollArea className="h-96">
              <div className="space-y-3">
                {events
                  .filter(event => event.impact === 'high')
                  .slice(0, 10)
                  .map((event) => (
                    <div key={event.id} className="p-3 border-2 border-orange-200 rounded-lg bg-orange-50">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-orange-600" />
                        <span className="font-medium text-orange-900">HIGH IMPACT</span>
                      </div>
                      {/* Same event display logic as recent tab */}
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getEventIcon(event.eventType)}
                          <span className="font-medium text-sm">{event.title}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {getConfidenceBadge(event.confidence)}
                          <Badge
                            variant="outline"
                            className={getSentimentColor(event.sentiment)}
                          >
                            {event.sentiment}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="watchlist" className="mt-4">
            <div className="text-center py-8 text-gray-500">
              <Settings className="w-8 h-8 mx-auto mb-2" />
              <p>Watchlist feature coming soon</p>
              <p className="text-sm">Configure symbols to track specific news events</p>
            </div>
          </TabsContent>
        </Tabs>
      </Card>

      {/* Event Details Modal */}
      {selectedEvent && (
        <Card className="p-4 border-2">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold">Event Details</h4>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedEvent(null)}
            >
              ×
            </Button>
          </div>

          <div className="space-y-4">
            <div>
              <h5 className="font-medium mb-2">{selectedEvent.title}</h5>
              <p className="text-sm text-gray-600">{selectedEvent.content}</p>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-sm font-medium">Classification</span>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="outline">{selectedEvent.classification.category}</Badge>
                  {selectedEvent.classification.subcategory && (
                    <Badge variant="secondary">{selectedEvent.classification.subcategory}</Badge>
                  )}
                  <Progress value={selectedEvent.classification.confidence * 100} className="w-16 h-2" />
                </div>
              </div>

              <div>
                <span className="text-sm font-medium">Impact Assessment</span>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant={selectedEvent.impact === 'high' ? 'destructive' : 'secondary'}>
                    {selectedEvent.impact.toUpperCase()}
                  </Badge>
                  <Progress value={selectedEvent.confidence * 100} className="w-16 h-2" />
                </div>
              </div>
            </div>

            {selectedEvent.reaction && (
              <>
                <Separator />
                <div>
                  <span className="text-sm font-medium">Expected Market Reaction</span>
                  <div className="grid grid-cols-3 gap-4 mt-2">
                    <div className="text-center">
                      <div className="text-lg font-bold text-blue-600">
                        {selectedEvent.reaction.expectedReturn > 0 ? '+' : ''}
                        {(selectedEvent.reaction.expectedReturn * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">Expected Return</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-green-600">
                        {(selectedEvent.reaction.confidence * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-500">Confidence</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-purple-600">
                        {selectedEvent.reaction.timeHorizon}
                      </div>
                      <div className="text-xs text-gray-500">Time Horizon</div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};
