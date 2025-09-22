import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Card } from '@/components/ui/Card';
import { useQuery } from '@tanstack/react-query';
import { 
  TrendingUp, 
  BarChart3, 
  Activity, 
  Target, 
  Zap, 
  Brain,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle
} from 'lucide-react';
import { useDecisionsRecent } from '@/hooks/useDecisionsRecent';
import { useSyncedStrategies, useSyncedOrders, useSyncedPipelineHealth, useSyncedEvoStatus } from '@/hooks/useSyncedData';
import { DATA_REFRESH_CONFIG } from '@/config/dataRefreshConfig';

const TradeDecisionsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'proposals');
  
  // Update URL when tab changes
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setSearchParams({ tab: value });
  };

  // Sync activeTab with URL params
  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && tab !== activeTab) {
      setActiveTab(tab);
    }
  }, [searchParams, activeTab]);

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Trade Decisions</h1>
        <p className="text-muted-foreground mt-2">
          Monitor strategies, pipeline processing, and execution tracking
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="strategies">Strategies</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="proposals">Proposals</TabsTrigger>
          <TabsTrigger value="intents">Intents</TabsTrigger>
          <TabsTrigger value="executions">Executions</TabsTrigger>
          <TabsTrigger value="evo">Evolution</TabsTrigger>
        </TabsList>

        <TabsContent value="strategies" className="mt-6">
          <StrategiesTab />
        </TabsContent>

        <TabsContent value="pipeline" className="mt-6">
          <PipelineTab />
        </TabsContent>

        <TabsContent value="proposals" className="mt-6">
          <ProposalsTab />
        </TabsContent>

        <TabsContent value="intents" className="mt-6">
          <IntentsTab />
        </TabsContent>

        <TabsContent value="executions" className="mt-6">
          <ExecutionsTab />
        </TabsContent>

        <TabsContent value="evo" className="mt-6">
          <EvolutionTab />
        </TabsContent>
      </Tabs>
    </div>
  );
};

// Strategies Tab
const StrategiesTab: React.FC = () => {
  const { data: strategiesData, isLoading } = useSyncedStrategies();
  const strategies = Array.isArray(strategiesData) ? strategiesData : strategiesData?.items || [];

  if (isLoading) {
    return <div className="text-center py-8">Loading strategies...</div>;
  }

  if (!strategies.length) {
    return (
      <Card className="p-8 text-center">
        <BarChart3 className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">No Active Strategies</h3>
        <p className="text-muted-foreground mt-2">Trading strategies will appear here when activated.</p>
      </Card>
    );
  }

  return (
    <div className="grid gap-4">
      {strategies.map((strategy: any) => (
        <Card key={strategy.id} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-lg font-semibold">{strategy.name || strategy.id}</h3>
              <p className="text-sm text-muted-foreground">{strategy.description || 'No description'}</p>
            </div>
            <span className={`px-2 py-1 rounded text-xs ${
              strategy.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
            }`}>
              {strategy.active ? 'Active' : 'Inactive'}
            </span>
          </div>
          
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Profit Factor</span>
              <p className="font-medium">{(strategy.performance?.profit_factor || 0).toFixed(2)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Sharpe Ratio</span>
              <p className="font-medium">{(strategy.performance?.sharpe_ratio || 0).toFixed(2)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Total Trades</span>
              <p className="font-medium">{strategy.performance?.trades_count || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Win Rate</span>
              <p className="font-medium">{((strategy.performance?.win_rate || 0) * 100).toFixed(1)}%</p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};

// Pipeline Tab
const PipelineTab: React.FC = () => {
  const { data: pipelineData, isLoading } = useSyncedPipelineHealth('15m');

  if (isLoading) {
    return <div className="text-center py-8">Loading pipeline data...</div>;
  }

  return (
    <div className="space-y-6">
      {pipelineData ? (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Pipeline Summary (15m)</h3>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <span className="text-muted-foreground text-sm">Total Processed</span>
              <p className="text-2xl font-bold">{pipelineData.total_scores || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Avg Score</span>
              <p className="text-2xl font-bold">{((pipelineData.avg_score || 0) * 100).toFixed(0)}%</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">High Confidence</span>
              <p className="text-2xl font-bold">{pipelineData.high_confidence || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Last Update</span>
              <p className="text-sm">{pipelineData.asOf ? new Date(pipelineData.asOf).toLocaleTimeString() : 'N/A'}</p>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="p-8 text-center">
          <Activity className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Pipeline Activity</h3>
          <p className="text-muted-foreground mt-2">Pipeline processing data will appear here.</p>
        </Card>
      )}
    </div>
  );
};

// Proposals Tab
const ProposalsTab: React.FC = () => {
  const { data: decisions, isLoading } = useDecisionsRecent(50);

  if (isLoading) {
    return <div className="text-center py-8">Loading proposals...</div>;
  }

  if (!decisions || decisions.length === 0) {
    return (
      <Card className="p-8 text-center">
        <Target className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">No Trade Proposals</h3>
        <p className="text-muted-foreground mt-2">Trade proposals will appear here as they are generated.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {decisions.map((d: any) => (
        <Card key={d.trace_id || d.id} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold">{d.symbol}</h3>
              <span className={`px-2 py-1 rounded text-xs ${
                d.action === 'BUY' || d.side === 'buy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {(d.action || d.side || '').toUpperCase()}
              </span>
            </div>
            <span className="text-sm text-muted-foreground">
              {new Date(d.createdAt || d.timestamp).toLocaleTimeString()}
            </span>
          </div>

          <p className="text-muted-foreground mb-4">{d.one_liner || d.reason || 'No reason provided'}</p>

          {d.plan && (
            <div className="p-3 bg-muted/30 rounded text-sm">
              <span className="font-medium">
                {d.plan.orderType || 'MARKET'} • Qty: {d.plan.qty || 0}
                {d.plan.limit && ` • Limit: $${d.plan.limit}`}
              </span>
            </div>
          )}
        </Card>
      ))}
    </div>
  );
};

// Intents Tab
const IntentsTab: React.FC = () => {
  const { data: decisions } = useDecisionsRecent(100);
  const intents = decisions?.filter((d: any) => 
    d.stage === 'intent' || 
    (d.gates?.passed === true && d.status !== 'executed')
  ) || [];

  if (!intents.length) {
    return (
      <Card className="p-8 text-center">
        <Clock className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">No Pending Intents</h3>
        <p className="text-muted-foreground mt-2">Approved trades awaiting execution will appear here.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {intents.map((d: any) => (
        <Card key={d.trace_id || d.id} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold">{d.symbol}</h3>
              <span className={`px-2 py-1 rounded text-xs ${
                d.action === 'BUY' || d.side === 'buy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {(d.action || d.side || '').toUpperCase()}
              </span>
              <span className="px-2 py-1 rounded text-xs bg-yellow-100 text-yellow-800">
                Pending
              </span>
            </div>
            <span className="text-sm text-muted-foreground">
              {new Date(d.createdAt || d.timestamp).toLocaleTimeString()}
            </span>
          </div>

          <p className="text-muted-foreground">{d.one_liner || d.reason || 'Awaiting execution'}</p>
        </Card>
      ))}
    </div>
  );
};

// Executions Tab
const ExecutionsTab: React.FC = () => {
  const { data: ordersData, isLoading } = useSyncedOrders();
  const orders = ordersData || [];

  if (isLoading) {
    return <div className="text-center py-8">Loading orders...</div>;
  }

  if (!orders.length) {
    return (
      <Card className="p-8 text-center">
        <Zap className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">No Orders</h3>
        <p className="text-muted-foreground mt-2">Order executions will appear here.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {orders.map((order: any) => (
        <Card key={order.id} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold">{order.symbol}</h3>
              <span className={`px-2 py-1 rounded text-xs ${
                order.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {order.side?.toUpperCase()}
              </span>
              <span className={`px-2 py-1 rounded text-xs ${
                order.status === 'filled' ? 'bg-green-100 text-green-800' :
                order.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {order.status?.toUpperCase()}
              </span>
            </div>
            <span className="text-sm text-muted-foreground">
              {new Date(order.created_at || order.timestamp).toLocaleTimeString()}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Type</span>
              <p className="font-medium">{order.type?.toUpperCase() || 'MARKET'}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Quantity</span>
              <p className="font-medium">{order.qty || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Price</span>
              <p className="font-medium">${order.price || order.limit || 'Market'}</p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};

// Evolution Tab
const EvolutionTab: React.FC = () => {
  const { data: evoData, isLoading } = useSyncedEvoStatus();

  if (isLoading) {
    return <div className="text-center py-8">Loading evolution data...</div>;
  }

  return (
    <div className="space-y-6">
      {evoData ? (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Evolution Status</h3>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <span className="text-muted-foreground text-sm">Generation</span>
              <p className="text-2xl font-bold">{evoData.generation || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Population</span>
              <p className="text-2xl font-bold">{evoData.population || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Best PF</span>
              <p className="text-2xl font-bold">{(evoData.best?.metrics?.pf_after_costs || 0).toFixed(2)}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Status</span>
              <p className={`text-sm font-medium ${
                evoData.running ? 'text-green-600' : 'text-gray-600'
              }`}>
                {evoData.running ? 'Running' : 'Stopped'}
              </p>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="p-8 text-center">
          <Brain className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Evolution Data</h3>
          <p className="text-muted-foreground mt-2">Evolution candidates will appear here.</p>
        </Card>
      )}
    </div>
  );
};

export default TradeDecisionsPage;