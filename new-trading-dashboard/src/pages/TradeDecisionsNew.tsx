import React, { useState, useEffect } from "react";
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import {
  Target,
  TrendingUp,
  Eye,
  RefreshCw,
  Calendar,
  Search,
  Download
} from 'lucide-react';
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";

export default function TradeDecisionsNewPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedSymbol] = useState<string>('SPY');
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidenceData, setEvidenceData] = useState<any>(null);
  const [timeframe, setTimeframe] = useState<'15m' | '1h' | '1d'>('15m');
  const [symbolFilter, setSymbolFilter] = useState<string>('');

  // Get initial tab from URL params or default based on mode
  const urlTab = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<string>(urlTab || 'proposals');

  // Auto-select default tab based on autoloop mode
  const { data: autoloopStatus } = useQuery({
    queryKey: ['audit', 'autoloop', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/audit/autoloop/status');
      if (!response.ok) throw new Error('Failed to fetch autoloop status');
      return response.json();
    },
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (autoloopStatus && !urlTab) {
      const mode = autoloopStatus.mode;
      if (mode === 'discovery') {
        setActiveTab('proposals');
      } else if (mode === 'shadow' || mode === 'live') {
        setActiveTab('intents');
      }
    }
  }, [autoloopStatus, urlTab]);

  // Update URL when tab changes
  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    setSearchParams({ tab });
  };

  // Data queries for each tab
  const { data: proposalsData } = useQuery({
    queryKey: ['decisions', 'recent', 'proposed', timeframe],
    queryFn: async () => {
      const response = await fetch(`/api/decisions/recent?stage=proposed&limit=200`);
      if (!response.ok) throw new Error('Failed to fetch proposals');
      const data = await response.json();
      return Array.isArray(data) ? data : data.items || [];
    },
    refetchInterval: 10000,
    enabled: activeTab === 'proposals',
  });

  const { data: intentsData } = useQuery({
    queryKey: ['decisions', 'recent', 'intent', timeframe],
    queryFn: async () => {
      const response = await fetch(`/api/decisions/recent?stage=intent&limit=200`);
      if (!response.ok) throw new Error('Failed to fetch intents');
      const data = await response.json();
      return Array.isArray(data) ? data : data.items || [];
    },
    refetchInterval: 10000,
    enabled: activeTab === 'intents',
  });

  const { data: executionsData } = useQuery({
    queryKey: ['paper', 'orders', 'full'],
    queryFn: async () => {
      const response = await fetch('/api/paper/orders?limit=200');
      if (!response.ok) throw new Error('Failed to fetch orders');
      const data = await response.json();
      return Array.isArray(data) ? data : data.items || [];
    },
    refetchInterval: 15000,
    enabled: activeTab === 'executions',
  });

  const { data: positionsData, isLoading: positionsLoading } = useQuery({
    queryKey: ['paper', 'positions'],
    queryFn: async () => {
      const response = await fetch('/api/paper/positions');
      if (!response.ok) throw new Error('Failed to fetch positions');
      const data = await response.json();
      return Array.isArray(data) ? data : data.items || [];
    },
    refetchInterval: 15000,
    enabled: activeTab === 'executions',
  });


  // Handle evidence viewing
  const handleViewEvidence = (data: any) => {
    setEvidenceData(data);
    setEvidenceOpen(true);
  };

  // Filter functions
  const filterBySymbol = (data: any[], symbol: string) => {
    return symbol ? data.filter(item => item.symbol === symbol) : data;
  };

  const filteredProposals = filterBySymbol(proposalsData || [], symbolFilter);
  const filteredIntents = filterBySymbol(intentsData || [], symbolFilter);

  return (
    <div className="w-full py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Target className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-white">Trade Decisions</h1>
            <p className="text-sm text-gray-400">Detailed audit trail and evidence for all trading decisions</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {autoloopStatus && (
            <Badge variant="outline" className="text-white border-blue-500">
              Mode: {autoloopStatus.mode}
            </Badge>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.location.reload()}
            className="text-xs"
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 p-4 bg-gray-900/50 rounded-lg">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Filter by symbol..."
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            className="px-3 py-1 bg-gray-800 border border-gray-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
          />
        </div>
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-gray-400" />
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value as '15m' | '1h' | '1d')}
            className="px-3 py-1 bg-gray-800 border border-gray-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
          >
            <option value="15m">15 minutes</option>
            <option value="1h">1 hour</option>
            <option value="1d">1 day</option>
          </select>
        </div>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid w-full grid-cols-9 bg-gray-800">
          <TabsTrigger value="proposals" className="text-xs">Proposals</TabsTrigger>
          <TabsTrigger value="intents" className="text-xs">Trade Intents</TabsTrigger>
          <TabsTrigger value="executions" className="text-xs">Executions</TabsTrigger>
          <TabsTrigger value="pipeline" className="text-xs">Pipeline</TabsTrigger>
          <TabsTrigger value="scoring" className="text-xs">Scoring</TabsTrigger>
          <TabsTrigger value="risk" className="text-xs">Risk Rejections</TabsTrigger>
          <TabsTrigger value="strategies" className="text-xs">Strategies</TabsTrigger>
          <TabsTrigger value="evo" className="text-xs">Evo</TabsTrigger>
          <TabsTrigger value="evidence" className="text-xs">Raw JSON</TabsTrigger>
        </TabsList>

        {/* Proposals Tab */}
        <TabsContent value="proposals" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Proposals</h2>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-cyan-400">
                  {filteredProposals.length} proposals
                </Badge>
                <Button variant="outline" size="sm" className="text-xs">
                  <Download className="w-3 h-3 mr-1" />
                  Export
                </Button>
              </div>
            </div>

            {proposalsLoading ? (
              <div className="text-center py-12">
                <div className="animate-pulse text-gray-400">Loading proposals...</div>
              </div>
            ) : filteredProposals.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No proposals found</p>
                <p className="text-sm">Proposals will appear here as the brain discovers opportunities</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-300">Time</th>
                      <th className="text-left py-3 px-4 text-gray-300">Symbol</th>
                      <th className="text-left py-3 px-4 text-gray-300">Strategy</th>
                      <th className="text-right py-3 px-4 text-gray-300">Confidence</th>
                      <th className="text-left py-3 px-4 text-gray-300">Reason</th>
                      <th className="text-center py-3 px-4 text-gray-300">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredProposals.map((proposal: any, idx: number) => (
                      <tr key={proposal.id || idx} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="py-3 px-4 text-gray-300">
                          {new Date(proposal.ts).toLocaleTimeString()}
                        </td>
                        <td className="py-3 px-4 font-medium text-white">{proposal.symbol}</td>
                        <td className="py-3 px-4 text-gray-300">{proposal.strategy_id}</td>
                        <td className="py-3 px-4 text-right">
                          <span className={`px-2 py-1 rounded text-xs ${
                            proposal.confidence > 0.7 ? 'bg-green-500/20 text-green-400' :
                            proposal.confidence > 0.5 ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-red-500/20 text-red-400'
                          }`}>
                            {(proposal.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-gray-400 max-w-xs truncate" title={proposal.reason}>
                          {proposal.reason}
                        </td>
                        <td className="py-3 px-4 text-center">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleViewEvidence(proposal)}
                            className="text-xs"
                          >
                            <Eye className="w-3 h-3 mr-1" />
                            Evidence
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </TabsContent>

        {/* Trade Intents Tab */}
        <TabsContent value="intents" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Trade Intents</h2>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-purple-400">
                  {filteredIntents.length} intents
                </Badge>
                <Button variant="outline" size="sm" className="text-xs">
                  <Download className="w-3 h-3 mr-1" />
                  Export
                </Button>
              </div>
            </div>

            {intentsLoading ? (
              <div className="text-center py-12">
                <div className="animate-pulse text-gray-400">Loading intents...</div>
              </div>
            ) : filteredIntents.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No trade intents found</p>
                <p className="text-sm">Trade intents appear after risk assessment</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-300">Time</th>
                      <th className="text-left py-3 px-4 text-gray-300">Symbol</th>
                      <th className="text-left py-3 px-4 text-gray-300">Side</th>
                      <th className="text-right py-3 px-4 text-gray-300">Qty</th>
                      <th className="text-right py-3 px-4 text-gray-300">Limit</th>
                      <th className="text-right py-3 px-4 text-gray-300">EV After Costs</th>
                      <th className="text-center py-3 px-4 text-gray-300">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredIntents.map((intent: any, idx: number) => (
                      <tr key={intent.id || idx} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="py-3 px-4 text-gray-300">
                          {new Date(intent.ts).toLocaleTimeString()}
                        </td>
                        <td className="py-3 px-4 font-medium text-white">{intent.symbol}</td>
                        <td className="py-3 px-4">
                          <span className={`px-2 py-1 rounded text-xs ${
                            intent.side === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {intent.side}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-gray-300">{intent.qty}</td>
                        <td className="py-3 px-4 text-right font-mono text-gray-300">{intent.limit}</td>
                        <td className="py-3 px-4 text-right">
                          <span className={`font-mono ${
                            intent.ev_after_costs > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {intent.ev_after_costs?.toFixed(4)}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleViewEvidence(intent)}
                            className="text-xs"
                          >
                            <Eye className="w-3 h-3 mr-1" />
                            Evidence
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </TabsContent>

        {/* Executions Tab */}
        <TabsContent value="executions" className="mt-6">
          <div className="space-y-6">
            {/* Orders Table */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Orders</h3>
                <Badge variant="outline" className="text-yellow-400">
                  {(executionsData || []).length} orders
                </Badge>
              </div>

              {executionsLoading ? (
                <div className="text-center py-8">
                  <div className="animate-pulse text-gray-400">Loading orders...</div>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-2 px-3 text-gray-300">Time</th>
                        <th className="text-left py-2 px-3 text-gray-300">Symbol</th>
                        <th className="text-left py-2 px-3 text-gray-300">Side</th>
                        <th className="text-right py-2 px-3 text-gray-300">Qty</th>
                        <th className="text-right py-2 px-3 text-gray-300">Price</th>
                        <th className="text-left py-2 px-3 text-gray-300">Status</th>
                        <th className="text-left py-2 px-3 text-gray-300">Strategy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(executionsData || []).map((order: any, idx: number) => (
                        <tr key={order.id || idx} className="border-b border-gray-800">
                          <td className="py-2 px-3 text-gray-300">
                            {new Date(order.ts || order.timestamp).toLocaleTimeString()}
                          </td>
                          <td className="py-2 px-3 font-medium text-white">{order.symbol}</td>
                          <td className="py-2 px-3">
                            <span className={`px-2 py-1 rounded text-xs ${
                              order.side === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                            }`}>
                              {order.side}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-right text-gray-300">{order.qty}</td>
                          <td className="py-2 px-3 text-right font-mono text-gray-300">
                            {order.price || order.limit}
                          </td>
                          <td className="py-2 px-3">
                            <span className={`px-2 py-1 rounded text-xs ${
                              order.status === 'FILLED' ? 'bg-green-500/20 text-green-400' :
                              order.status === 'PENDING' ? 'bg-yellow-500/20 text-yellow-400' :
                              'bg-red-500/20 text-red-400'
                            }`}>
                              {order.status}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-gray-400">{order.strategy_id}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Positions Table */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Positions</h3>
                <Badge variant="outline" className="text-blue-400">
                  {(positionsData || []).length} positions
                </Badge>
              </div>

              {positionsLoading ? (
                <div className="text-center py-8">
                  <div className="animate-pulse text-gray-400">Loading positions...</div>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left py-2 px-3 text-gray-300">Symbol</th>
                        <th className="text-right py-2 px-3 text-gray-300">Qty</th>
                        <th className="text-right py-2 px-3 text-gray-300">Avg Price</th>
                        <th className="text-right py-2 px-3 text-gray-300">Current</th>
                        <th className="text-right py-2 px-3 text-gray-300">PnL</th>
                        <th className="text-right py-2 px-3 text-gray-300">PnL %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(positionsData || []).map((position: any, idx: number) => (
                        <tr key={position.symbol || idx} className="border-b border-gray-800">
                          <td className="py-2 px-3 font-medium text-white">{position.symbol}</td>
                          <td className="py-2 px-3 text-right text-gray-300">{position.qty}</td>
                          <td className="py-2 px-3 text-right font-mono text-gray-300">
                            {position.avg_price?.toFixed(2)}
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-gray-300">
                            {position.current_price?.toFixed(2)}
                          </td>
                          <td className={`py-2 px-3 text-right font-mono ${
                            (position.unrealized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {position.unrealized_pnl?.toFixed(2)}
                          </td>
                          <td className={`py-2 px-3 text-right font-mono ${
                            (position.unrealized_pnl_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {((position.unrealized_pnl_pct || 0) * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        {/* Raw JSON Evidence Tab */}
        <TabsContent value="evidence" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Raw JSON Evidence</h2>
              <Badge variant="outline" className="text-gray-400">
                Evidence Viewer
              </Badge>
            </div>

            <div className="border rounded-lg p-4 bg-gray-900/50">
              <p className="text-gray-400 text-center py-12">
                Select "Evidence" from any row in other tabs to view raw JSON data here
              </p>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* Evidence Drawer */}
      {evidenceOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border rounded-lg max-w-4xl w-full mx-4 max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold text-white">Raw JSON Evidence</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setEvidenceOpen(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </Button>
            </div>
            <div className="p-4 overflow-auto max-h-[60vh]">
              <pre className="text-xs text-gray-300 whitespace-pre-wrap">
                {JSON.stringify(evidenceData, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
