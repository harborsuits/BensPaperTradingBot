import React, { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  ArrowUpDown, 
  TrendingUp, 
  TrendingDown, 
  CheckCircle, 
  XCircle,
  BarChart2,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';

import { decisionApi } from '@/services/api';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { TradeCandidate } from '@/types/api.types';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Progress } from '@/components/ui/Progress';
import { Badge } from '@/components/ui/Badge';
import WebSocketIndicator from '@/components/ui/WebSocketIndicator';

const TradeCandidatesPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [candidateFilter, setCandidateFilter] = useState<string>('all'); // 'all', 'executed', 'skipped'
  const [strategyFilter, setStrategyFilter] = useState<string>('all');
  const [lastUpdateTimestamp, setLastUpdateTimestamp] = useState<Date | null>(null);
  
  // Connect to WebSocket for real-time updates
  const { isConnected } = useWebSocketChannel('decisions', true);
  
  // Fetch latest trade candidates
  const { 
    data: candidates = [] as TradeCandidate[],
    isLoading,
    isError,
    error
  } = useQuery({
    queryKey: ['trade-candidates'],
    queryFn: async () => {
      const response = await decisionApi.getLatestDecisions();
      return response.success ? response.data : [];
    },
    staleTime: 30000 // 30 seconds
  });
  
  // Handle WebSocket updates for new trade decisions
  useWebSocketMessage<{candidates: TradeCandidate[]}>
  (
    'cycle_decision',
    (message) => {
      if (message.data && message.data.candidates) {
        // Update the candidates cache when new decisions come in
        queryClient.setQueryData(['trade-candidates'], message.data.candidates);
        setLastUpdateTimestamp(new Date());
      }
    }
  );
  
  // Get unique strategies from candidates
  const uniqueStrategies = React.useMemo(() => {
    const strategies = new Set<string>();
    candidates.forEach(candidate => {
      if (candidate.strategy_name) {
        strategies.add(candidate.strategy_name);
      }
    });
    return Array.from(strategies);
  }, [candidates]);
  
  // Filter candidates based on current filters
  const filteredCandidates = React.useMemo(() => {
    return candidates.filter(candidate => {
      // Filter by execution status
      if (candidateFilter === 'executed' && !candidate.executed) {
        return false;
      }
      if (candidateFilter === 'skipped' && candidate.executed) {
        return false;
      }
      // Filter by strategy
      if (strategyFilter !== 'all' && candidate.strategy_name !== strategyFilter) {
        return false;
      }
      return true;
    });
  }, [candidates, candidateFilter, strategyFilter]);
  
  const getDirectionIcon = (direction: 'buy' | 'sell') => {
    return direction === 'buy' 
      ? <TrendingUp className="text-bull" size={16} /> 
      : <TrendingDown className="text-bear" size={16} />;
  };
  
  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-success';
    if (score >= 0.5) return 'text-primary';
    if (score >= 0.3) return 'text-amber-500';
    return 'text-muted-foreground';
  };
  
    // Get color for score progress bar
  const getScoreBarColor = (score: number) => {
    if (score >= 0.7) return 'bg-success';
    if (score >= 0.5) return 'bg-primary';
    if (score >= 0.3) return 'bg-amber-500';
    return 'bg-muted-foreground';
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-2 md:space-y-0">
        <div>
          <h1 className="text-2xl font-bold">Trade Candidates</h1>
          <p className="text-muted-foreground">View all trade candidates evaluated in the last cycle</p>
        </div>
        <div className="flex items-center gap-4">
          <WebSocketIndicator className="text-xs" />
          <div className="text-xs text-muted-foreground hidden md:block">
            {lastUpdateTimestamp ? (
              <>Last update: {lastUpdateTimestamp.toLocaleTimeString()}</>
            ) : (
              isConnected ? 'Waiting for decisions...' : 'Connecting...'
            )}
          </div>
          <Button
            onClick={() => {
              queryClient.invalidateQueries(['trade-candidates']);
            }}
            variant="outline"
            size="sm"
            className="flex items-center gap-1"
          >
            <RefreshCw size={14} />
            Refresh
          </Button>
        </div>
      </div>
      
      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <div className="flex bg-muted/30 rounded-md overflow-hidden">
          <button
            onClick={() => setCandidateFilter('all')}
            className={`px-4 py-2 text-sm ${
              candidateFilter === 'all' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
            }`}
          >
            All Candidates
          </button>
          <button
            onClick={() => setCandidateFilter('executed')}
            className={`px-4 py-2 text-sm ${
              candidateFilter === 'executed' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
            }`}
          >
            Executed Only
          </button>
          <button
            onClick={() => setCandidateFilter('skipped')}
            className={`px-4 py-2 text-sm ${
              candidateFilter === 'skipped' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
            }`}
          >
            Skipped Only
          </button>
        </div>
        
        {uniqueStrategies.length > 0 && (
          <div className="flex">
            <select
              className="h-9 w-36 rounded-md border border-border bg-background px-3 text-sm"
              value={strategyFilter}
              onChange={(e) => setStrategyFilter(e.target.value)}
              aria-label="Filter by strategy"
            >
              <option value="all">All Strategies</option>
              {uniqueStrategies.map(strategy => (
                <option key={strategy} value={strategy}>{strategy}</option>
              ))}
            </select>
          </div>
        )}
      </div>
      
      {/* Candidates List */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center">
            <BarChart2 size={18} className="mr-2" />
            Trade Decision Candidates
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-24 rounded-md bg-muted/20 animate-pulse" />
              ))}
            </div>
          ) : isError ? (
            <div className="py-8 flex flex-col items-center justify-center">
              <AlertTriangle size={32} className="text-destructive mb-2" />
              <p className="text-destructive font-medium">Error loading trade candidates</p>
              <p className="text-muted-foreground text-sm mt-1">
                {error instanceof Error ? error.message : 'Unknown error'}
              </p>
            </div>
          ) : filteredCandidates.length === 0 ? (
            <div className="py-12 flex flex-col items-center justify-center">
              <ArrowUpDown size={32} className="text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No trade candidates match your filters</p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredCandidates.map((candidate) => (
                <div 
                  key={`${candidate.symbol}-${candidate.strategy_id}`}
                  className={`border rounded-lg p-4 ${
                    candidate.executed 
                      ? 'border-success/30 bg-success/5' 
                      : 'border-muted'
                  }`}
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center">
                      {getDirectionIcon(candidate.direction)}
                      <span className={`text-lg font-semibold ml-2 ${
                        candidate.direction === 'buy' ? 'text-bull' : 'text-bear'
                      }`}>
                        {candidate.symbol}
                      </span>
                      {candidate.executed ? (
                        <Badge variant="success" className="ml-2">Executed</Badge>
                      ) : (
                        <Badge variant="outline" className="ml-2">Skipped</Badge>
                      )}
                    </div>
                    <div className="flex items-center">
                      <Badge variant="outline" className="mr-2">
                        {candidate.strategy_name || 'Unknown Strategy'}
                      </Badge>
                      {candidate.executed ? (
                        <CheckCircle size={18} className="text-success" />
                      ) : (
                        <XCircle size={18} className="text-muted-foreground" />
                      )}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-muted-foreground">Signal Score</span>
                        <span className={getScoreColor(candidate.score)}>
                          {(candidate.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <Progress 
                        value={candidate.score * 100} 
                        className={`h-2 ${getScoreBarColor(candidate.score)}`}
                      />
                    </div>
                    
                    <div className="flex flex-col">
                      <span className="text-sm text-muted-foreground mb-1">Entry Price</span>
                      <span className="font-medium">${candidate.entry_price.toFixed(2)}</span>
                    </div>
                    
                    <div className="flex flex-col">
                      <span className="text-sm text-muted-foreground mb-1">Risk/Reward</span>
                      <span className="font-medium">{candidate.risk_reward_ratio.toFixed(2)}</span>
                    </div>
                  </div>
                  
                  {candidate.entry_conditions?.length > 0 && (
                    <div className="mt-2">
                      <div className="text-xs text-muted-foreground mb-1">Entry Conditions:</div>
                      <div className="flex flex-wrap gap-1">
                        {candidate.entry_conditions.map((condition, index) => (
                          <div 
                            key={index} 
                            className="text-xs px-2 py-0.5 rounded-full bg-muted/30"
                          >
                            {condition}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {candidate.reason && (
                    <div className="mt-2 pt-2 border-t border-border">
                      <div className="text-xs text-muted-foreground mb-1">Decision Rationale:</div>
                      <p className="text-sm">
                        {candidate.executed 
                          ? candidate.reason 
                          : candidate.reason || "Did not meet execution criteria"}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TradeCandidatesPage;
