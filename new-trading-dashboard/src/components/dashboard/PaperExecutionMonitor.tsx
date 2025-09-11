import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Clock,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  ExternalLink,
  BarChart3
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

interface PaperOrder {
  order_id: string;
  decision_id: string | null;
  symbol: string;
  side: string;
  qty: number;
  px_req: number | null;
  px_fill: number | null;
  slippage_bps: number | null;
  status: string;
  t_create: string;
  t_fill: string | null;
  exec_ms: number | null;
  fail_reason: string | null;
  broker_order_id: string | null;
}

interface OrderMetrics {
  fill_rate: number;
  avg_slippage_bps: number;
  avg_exec_ms: number;
}

interface OrdersResponse {
  items: PaperOrder[];
  metrics: OrderMetrics;
  meta: {
    asOf: string;
    source: string;
    total: number;
    schema_version: string;
  };
}

const PaperExecutionMonitor: React.FC = () => {
  const [activeTab, setActiveTab] = useState('pending');
  const [liveOrders, setLiveOrders] = useState<PaperOrder[]>([]);

  const { data, isLoading, error, refetch } = useQuery<OrdersResponse>({
    queryKey: ['paper-orders', activeTab],
    queryFn: async () => {
      const response = await fetch(`/api/paper/orders?limit=100`);
      if (!response.ok) throw new Error('Failed to fetch paper orders');
      return response.json();
    },
    refetchInterval: 5000, // Refresh every 5 seconds
    staleTime: 2000,
  });

  const orders = data?.items || [];
  const metrics = data?.metrics;

  // SSE for live updates
  useEffect(() => {
    const eventSource = new EventSource('/api/paper/orders/stream');

    eventSource.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        if (update.type === 'order_update') {
          setLiveOrders(prev => {
            const existing = prev.find(o => o.order_id === update.data.order_id);
            if (existing) {
              return prev.map(o => o.order_id === update.data.order_id ? update.data : o);
            } else {
              return [update.data, ...prev].slice(0, 50);
            }
          });
        }
      } catch (e) {
        console.error('Failed to parse SSE message:', e);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const formatTime = (isoString: string) => {
    return new Date(isoString).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDuration = (ms: number | null) => {
    if (!ms) return '--';
    if (ms < 1000) return `${ms}ms`;
    return `${Math.round(ms / 1000)}s`;
  };

  const formatPrice = (price: number | null) => {
    return price ? `$${price.toFixed(2)}` : '--';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'bg-green-100 text-green-800';
      case 'working': return 'bg-blue-100 text-blue-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'failed': return 'bg-red-100 text-red-800';
      case 'cancelled': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled': return <CheckCircle className="w-3 h-3" />;
      case 'working': return <Loader2 className="w-3 h-3 animate-spin" />;
      case 'failed': return <XCircle className="w-3 h-3" />;
      case 'pending': return <Clock className="w-3 h-3" />;
      default: return <Clock className="w-3 h-3" />;
    }
  };

  const getSlippageColor = (slippage: number | null) => {
    if (!slippage) return 'text-gray-500';
    const absSlippage = Math.abs(slippage);
    if (absSlippage > 50) return 'text-red-600';
    if (absSlippage > 20) return 'text-yellow-600';
    return 'text-green-600';
  };

  const openTrace = (decisionId: string | null) => {
    if (decisionId) {
      window.open(`/decisions?trace=${decisionId}`, '_blank');
    }
  };

  if (isLoading && !data) {
    return (
      <div className="border rounded-xl p-4 w-full max-w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">📊 Paper Execution Monitor</h3>
          <div className="text-xs text-muted-foreground">Loading...</div>
        </div>
        <div className="animate-pulse space-y-4">
          <div className="h-16 bg-gray-200 rounded"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="border rounded-xl p-4 w-full max-w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">📊 Paper Execution Monitor</h3>
          <div className="text-xs text-red-500">Error loading orders</div>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <AlertTriangle className="mx-auto h-8 w-8 mb-2" />
          <p>Unable to load paper orders data</p>
          <button
            onClick={() => refetch()}
            className="mt-2 px-3 py-1 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="border rounded-xl p-4 w-full max-w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">📊 Paper Execution Monitor</h3>
        <div className="text-xs text-muted-foreground">
          {orders.length} orders • auto-refresh
        </div>
      </div>

      {/* KPIs */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium">Fill Rate</span>
            </div>
            <div className="text-2xl font-bold text-blue-600">
              {(metrics.fill_rate * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">Last 4h</div>
          </div>

          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium">Avg Slippage</span>
            </div>
            <div className="text-2xl font-bold text-green-600">
              {metrics.avg_slippage_bps}bps
            </div>
            <div className="text-xs text-muted-foreground">Filled orders</div>
          </div>

          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-purple-600" />
              <span className="text-sm font-medium">Avg Exec Time</span>
            </div>
            <div className="text-2xl font-bold text-purple-600">
              {formatDuration(metrics.avg_exec_ms)}
            </div>
            <div className="text-xs text-muted-foreground">Fill to execution</div>
          </div>
        </div>
      )}

      {/* Orders Table */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="pending">Pending</TabsTrigger>
          <TabsTrigger value="working">Working</TabsTrigger>
          <TabsTrigger value="filled">Filled</TabsTrigger>
          <TabsTrigger value="failed">Failed</TabsTrigger>
        </TabsList>

        {['pending', 'working', 'filled', 'failed'].map((status) => (
          <TabsContent key={status} value={status} className="mt-4">
            <div className="border rounded-lg overflow-hidden">
              <div className="max-h-96 overflow-y-auto">
                {orders.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Clock className="mx-auto h-8 w-8 mb-2 opacity-50" />
                    <p>No {status} orders in the last 4 hours</p>
                  </div>
                ) : (
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        <th className="text-left p-3 font-medium">Time</th>
                        <th className="text-left p-3 font-medium">Symbol</th>
                        <th className="text-left p-3 font-medium">Side</th>
                        <th className="text-right p-3 font-medium">Qty</th>
                        <th className="text-right p-3 font-medium">Req Price</th>
                        <th className="text-right p-3 font-medium">Fill Price</th>
                        <th className="text-right p-3 font-medium">Slippage</th>
                        <th className="text-right p-3 font-medium">Exec Time</th>
                        <th className="text-center p-3 font-medium">Status</th>
                        <th className="text-center p-3 font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {orders.map((order) => (
                        <tr key={order.order_id} className="border-t hover:bg-muted/30">
                          <td className="p-3 text-muted-foreground">
                            {formatTime(order.t_create)}
                          </td>
                          <td className="p-3 font-medium">{order.symbol}</td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              order.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                            }`}>
                              {order.side?.toUpperCase()}
                            </span>
                          </td>
                          <td className="p-3 text-right">{order.qty}</td>
                          <td className="p-3 text-right">{formatPrice(order.px_req)}</td>
                          <td className="p-3 text-right">{formatPrice(order.px_fill)}</td>
                          <td className="p-3 text-right">
                            <span className={getSlippageColor(order.slippage_bps)}>
                              {order.slippage_bps ? `${order.slippage_bps}bps` : '--'}
                            </span>
                          </td>
                          <td className="p-3 text-right">{formatDuration(order.exec_ms)}</td>
                          <td className="p-3 text-center">
                            <Badge variant="outline" className={getStatusColor(order.status)}>
                              <div className="flex items-center gap-1">
                                {getStatusIcon(order.status)}
                                <span className="capitalize">{order.status}</span>
                              </div>
                            </Badge>
                          </td>
                          <td className="p-3 text-center">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => openTrace(order.decision_id)}
                              disabled={!order.decision_id}
                              className="h-6 w-6 p-0"
                            >
                              <ExternalLink className="h-3 w-3" />
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default PaperExecutionMonitor;
