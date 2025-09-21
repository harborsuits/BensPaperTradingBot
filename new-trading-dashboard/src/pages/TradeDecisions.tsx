// src/pages/TradeDecisions.tsx
import React, { useState, useEffect } from "react";
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import {
  Target,
  TrendingUp,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Eye,
  RefreshCw,
  Filter,
  Calendar,
  Search,
  Download
} from 'lucide-react';
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { j, asItems } from "@/lib/api";

// TODO: Import missing components and utilities when they are implemented
// import DecisionCard from "@/components/decisions/DecisionCard";
// import EvidenceDrawer from "@/components/evidence/EvidenceDrawer";
// import { buildEvidenceFromUi, enrichWithWhy } from "@/lib/evidence";

type DecisionStage = 'proposed' | 'intent' | 'executed';

export default function TradeDecisionsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedSymbol, setSelectedSymbol] = useState<string>('SPY');
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidenceData, setEvidenceData] = useState<any>(null);
  const [evidencePacket, setEvidencePacket] = useState<any>(null);
  const [timeframe, setTimeframe] = useState<'15m' | '1h' | '1d'>('15m');
  const [symbolFilter, setSymbolFilter] = useState<string>('');

  // Get initial tab from URL params or default based on mode
  const urlTab = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<string>(urlTab || 'proposals');

  // Add missing state variables
  const [activeStage, setActiveStage] = useState<DecisionStage>('proposed');
  const [status, setStatus] = useState<string>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [invalid, setInvalid] = useState<number>(0);
  const [stageLoading, setStageLoading] = useState<boolean>(false);

  // Auto-select default tab based on autoloop mode
  const { data: autoloopStatus } = useQuery({
    queryKey: ['audit', 'autoloop', 'status'],
    queryFn: () => j('/api/audit/autoloop/status'),
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

  // Handle stage change for the decision stages
  const handleStageChange = (stage: DecisionStage) => {
    setActiveStage(stage);
  };

  // Data queries for each tab
  const { data: proposalsData } = useQuery({
    queryKey: ['decisions:proposed'],
    queryFn: () => j('/api/decisions/recent?stage=proposed&limit=200'),
    refetchInterval: 2500,
    enabled: activeTab === 'proposals',
  });

  const { data: intentsData } = useQuery({
    queryKey: ['decisions:intent'],
    queryFn: () => j('/api/decisions/recent?stage=intent&limit=200'),
    refetchInterval: 2500,
    enabled: activeTab === 'intents',
  });

  const { data: executionsData } = useQuery({
    queryKey: ['paper:orders'],
    queryFn: () => j('/api/paper/orders?limit=200'),
    refetchInterval: 5000,
    enabled: activeTab === 'executions',
  });







  // Get current decisions based on active tab
  const decisions = activeTab === 'proposals' ? asItems(proposalsData) :
                   activeTab === 'intents' ? asItems(intentsData) :
                   activeTab === 'executions' ? asItems(executionsData) : [];

  // Handle evidence viewing
  const handleViewEvidence = (data: any) => {
    setEvidenceData(data);
    setEvidenceOpen(true);
  };

  // Filter by symbol if selected
  const filteredDecisions = selectedSymbol && selectedSymbol !== 'SPY'
    ? decisions.filter(d => d.symbol === selectedSymbol)
    : decisions;

  // Get unique symbols for filter
  const symbols = Array.from(new Set(decisions.map(d => d.symbol || 'SPY')));

  // Handle opening evidence drawer
  const handleOpenEvidence = (d: any) => {
    try {
      // Convert DecisionTrace to EvidencePacket format
      const baseEvidence = {
        decision: {
          symbol: d.symbol,
          trace_id: d.trace_id,
          score: d.candidate_score?.alpha ? Math.round(d.candidate_score.alpha * 100) : undefined,
          reason: d.explain_layman,
          strategy: d.plan?.strategyLabel,
          createdAt: d.as_of,
        },
        context: d.market_context ? {
          regime: d.market_context.regime?.label,
          vix: d.market_context.volatility?.vix,
          bias: d.market_context.sentiment?.label,
        } : undefined,
      };

      // For now, just set the evidence packet directly
      // TODO: Implement proper buildEvidenceFromUi and enrichWithWhy functions
      setEvidencePacket(baseEvidence);
      setEvidenceOpen(true);
    } catch (e) {
      console.error("Failed to build evidence packet:", e);
      // Fallback: just set basic data
      setEvidencePacket({
        decision: {
          symbol: d.symbol || 'SPY',
          reason: 'Decision details',
          createdAt: new Date().toISOString(),
        }
      });
      setEvidenceOpen(true);
    }
  };

  // Temporary placeholder components
  const DecisionCard = ({ d, onOpenEvidence }: { d: any, onOpenEvidence: (d: any) => void }) => (
    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-white font-semibold">{d.symbol || 'SPY'}</h3>
          <p className="text-gray-400 text-sm">{d.strategy_id || 'Unknown Strategy'}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-green-400">{d.confidence ? `${Math.round(d.confidence * 100)}%` : 'N/A'}</span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => onOpenEvidence(d)}
            className="text-blue-400 border-blue-400 hover:bg-blue-400 hover:text-white"
          >
            <Eye className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );

  const EvidenceDrawer = ({ open, onOpenChange, data }: { open: boolean, onOpenChange: (open: boolean) => void, data: any }) => (
    <div className={`fixed right-0 top-0 h-full w-96 bg-gray-900 border-l border-gray-700 transform transition-transform ${open ? 'translate-x-0' : 'translate-x-full'}`}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-white text-lg font-semibold">Evidence Details</h2>
          <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
            <XCircle className="w-5 h-5" />
          </Button>
        </div>
        {data && (
          <div className="space-y-4">
            <div>
              <label className="text-gray-400 text-sm">Symbol</label>
              <p className="text-white">{data.decision?.symbol || 'N/A'}</p>
            </div>
            <div>
              <label className="text-gray-400 text-sm">Reason</label>
              <p className="text-white">{data.decision?.reason || 'No reason provided'}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="w-full py-6 space-y-6">
      {/* Header with status */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-white">Trade Decisions</h1>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-white border-blue-500">
            Connected
          </Badge>
          {autoloopStatus && (
            <Badge variant="outline" className="text-white border-blue-500">
              Mode: {autoloopStatus.mode}
            </Badge>
          )}
        </div>
      </div>

      {/* Stage tabs */}
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-gray-800">
          <TabsTrigger
            value="proposals"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Proposals
          </TabsTrigger>
          <TabsTrigger
            value="intents"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Trade Intents
          </TabsTrigger>
          <TabsTrigger
            value="executions"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Executions
          </TabsTrigger>
        </TabsList>

        {/* Symbol filter - shown for all tabs */}
        <div className="flex flex-wrap gap-2 mt-4">
          <Badge
            className="cursor-pointer bg-blue-800 hover:bg-blue-700 text-white"
            variant={selectedSymbol === null ? "default" : "outline"}
            onClick={() => setSelectedSymbol(null)}
          >
            All
          </Badge>
          {symbols.map(symbol => (
            <Badge
              key={symbol}
              className="cursor-pointer bg-blue-800 hover:bg-blue-700 text-white"
              variant={selectedSymbol === symbol ? "default" : "outline"}
              onClick={() => setSelectedSymbol(symbol)}
            >
              {symbol}
            </Badge>
          ))}
        </div>

        {/* Content for each tab */}
        <TabsContent value="proposals" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredDecisions.length === 0 ? (
              <div className="col-span-full text-center py-10 text-gray-400">
                {autoloopStatus?.mode === 'discovery' ? (
                  <div>
                    <p className="text-lg mb-2">No proposals yet</p>
                    <p className="text-sm">Proposals will appear here as the brain discovers trading opportunities.</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-lg mb-2">No proposals found</p>
                    <p className="text-sm">Switch to Trade Intents or Executions for current activity.</p>
                  </div>
                )}
              </div>
            ) : (
              filteredDecisions.map(decision => (
                <DecisionCard
                  key={decision.trace_id || decision.id}
                  d={decision}
                  onOpenEvidence={handleOpenEvidence}
                />
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="intents" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredDecisions.length === 0 ? (
              <div className="col-span-full text-center py-10 text-gray-400">
                <div>
                  <p className="text-lg mb-2">No trade intents</p>
                  <p className="text-sm">
                    Trade intents appear here after risk assessment and planning.
                    {(autoloopStatus?.mode === 'discovery') && " Enable Shadow or Live mode to see intents."}
                  </p>
                </div>
              </div>
            ) : (
              filteredDecisions.map(decision => (
                <DecisionCard
                  key={decision.trace_id || decision.id}
                  d={decision}
                  onOpenEvidence={handleOpenEvidence}
                />
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="executions" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredDecisions.length === 0 ? (
              <div className="col-span-full text-center py-10 text-gray-400">
                <div>
                  <p className="text-lg mb-2">No executions</p>
                  <p className="text-sm">Executed trades will appear here from the broker/OMS system.</p>
                </div>
              </div>
            ) : (
              filteredDecisions.map(decision => (
                <DecisionCard
                  key={decision.trace_id || decision.id}
                  d={decision}
                  onOpenEvidence={handleOpenEvidence}
                />
              ))
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* Evidence drawer */}
      <EvidenceDrawer
        open={evidenceOpen}
        onOpenChange={setEvidenceOpen}
        data={evidencePacket}
      />
    </div>
  );
}

