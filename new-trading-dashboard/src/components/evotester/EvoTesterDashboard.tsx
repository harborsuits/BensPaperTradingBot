/**
 * ============================================
 * EVO TESTER DASHBOARD - MAIN CONTAINER
 * ============================================
 * [1] Evolution Status Bar - Top status overview
 * [2] System Functionality Demo - Technical details
 * [3] Evolution Progress Section - Charts, controls, session management
 * [4] Active Sessions & History - Running experiments and session history
 * [5] Research & Discovery Hub - News analysis, fundamental research, strategy hypotheses, market discovery
 * [6] Evolution Results Hub - Strategy results, explanation, pipeline, deployment
 * [7] Evolution Lifecycle View - Timeline, champion lineage, population dynamics
 * [8] Evolution Sandbox - Auto-triggers, capital management, automated experiments
 * [9] Promotion Pipeline - Strategy promotion criteria, validation gates
 */

import React, { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  Play, Pause, StopCircle, RefreshCw, Save, Award, 
  ArrowUpRight, AlertTriangle
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import TimeSeriesChart from '@/components/ui/TimeSeriesChart';
import { Button } from '@/components/ui/Button';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { evoTesterApi } from '@/services/api';
import { showSuccessToast, showErrorToast } from '@/utils/toast.js';
import useEvoTesterWebSocket from '@/hooks/useEvoTesterWebSocket';
import { useEvoTesterUpdates } from '@/hooks/useEvoTesterUpdates';
import { EvoStrategy, EvoTesterConfig } from '@/types/api.types';
// Import components with explicit file extensions to help TypeScript resolution
import FitnessTrendChart from './FitnessTrendChart';
import StrategyParametersView from './StrategyParametersView';
import ActiveSessionsList from './ActiveSessionsList';
import EvolutionStatusBar from './EvolutionStatusBar';
import EvoLifecycleView from './EvoLifecycleView';
import EvolutionSandbox from './EvolutionSandbox';
import PromotionPipeline from './PromotionPipeline';
import { EvolutionResultsHub } from './EvolutionResultsHub';
import { ResearchDiscoveryHub } from './ResearchDiscoveryHub';
import styles from './EvoTesterDashboard.module.css';

interface EvoTesterDashboardProps {
  className?: string;
}

const EvoTesterDashboard: React.FC<EvoTesterDashboardProps> = ({ className = '' }) => {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'active' | 'history'>('active');
  const [selectedStrategy, setSelectedStrategy] = useState<EvoStrategy | null>(null);

  const queryClient = useQueryClient();

  // Setup WebSocket updates for notifications and live updates
  useEvoTesterWebSocket(activeSessionId || undefined);

  // Coordinated refresh strategy for all EvoTester components:
  // Fast (10-20s): Pipeline health, session status, real-time decisions/trades
  // Medium (30-60s): Market context, portfolio, active strategies
  // Slow (120-300s): Historical data, evolution history, performance metrics
  
  // Get real-time updates for active session
  const { 
    isRunning, 
    progress, 
    generations, 
    result, 
    error, 
    startEvoTest, 
    stopEvoTest 
  } = useEvoTesterUpdates(activeSessionId || undefined);

  // Fetch active EvoTester sessions from API
  const { data: activeSessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['evoTester', 'sessions'],
    queryFn: async () => {
      const response = await evoTesterApi.getEvoHistory();
      // Filter only recent sessions that are still running
      if (response.success && response.data) {
        return response.data.filter(session => {
          // Consider sessions from the last 24 hours
          const sessionDate = new Date(session.date);
          const oneDayAgo = new Date();
          oneDayAgo.setDate(oneDayAgo.getDate() - 1);
          return sessionDate > oneDayAgo;
        });
      }
      return [];
    },
    refetchInterval: 60000, // Refetch every minute
  });

  // Fetch history of completed sessions
  const { data: sessionHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['evoTester', 'history'],
    queryFn: async () => {
      const response = await evoTesterApi.getEvoHistory();
      return response.success ? response.data : [];
    },
    enabled: viewMode === 'history',
  });

  // Load session details when a session is selected
  useEffect(() => {
    if (activeSessionId) {
      const fetchSessionDetails = async () => {
        try {
          const response = await evoTesterApi.getEvoStatus(activeSessionId);
          if (response.success && response.data) {
            // Cache the response for future reference
            queryClient.setQueryData(['evoTester', 'status', activeSessionId], response.data);
          }
        } catch (err) {
          console.error('Error fetching session details:', err);
        }
      };
      
      fetchSessionDetails();
    }
  }, [activeSessionId, queryClient]);

  // Handle starting a new evolution session
  const handleStartNewSession = async (config: EvoTesterConfig) => {
    try {
      const newSessionId = await startEvoTest(config);
      if (newSessionId) {
        setActiveSessionId(newSessionId);
        showSuccessToast('Evolution session started successfully!');
        // Invalidate the sessions query to refresh the list
        queryClient.invalidateQueries(['evoTester', 'sessions']);
      }
    } catch (err) {
      console.error('Failed to start evolution session:', err);
      showErrorToast('Failed to start evolution session');
    }
  };

  // Handle stopping the current evolution session
  const handleStopSession = async () => {
    if (!activeSessionId) return;
    
    try {
      const success = await stopEvoTest(activeSessionId);
      if (success) {
        showSuccessToast('Evolution session stopped');
        // Invalidate the sessions query to refresh the list
        queryClient.invalidateQueries(['evoTester', 'sessions']);
      } else {
        showErrorToast('Failed to stop evolution session');
      }
    } catch (err) {
      console.error('Error stopping evolution session:', err);
      showErrorToast('Error stopping evolution session');
    }
  };

  // Handle pausing the current evolution session
  const handlePauseSession = async () => {
    if (!activeSessionId) return;
    
    try {
      const response = await evoTesterApi.pauseEvoTest(activeSessionId);
      if (response.success) {
        showSuccessToast('Evolution session paused');
        queryClient.invalidateQueries(['evoTester', 'status', activeSessionId]);
      } else {
        showErrorToast('Failed to pause evolution session');
      }
    } catch (err) {
      console.error('Error pausing evolution session:', err);
      showErrorToast('Error pausing evolution session');
    }
  };

  // Handle resuming a paused evolution session
  const handleResumeSession = async () => {
    if (!activeSessionId) {
      showErrorToast('No active session to resume');
      return;
    }

    try {
      // Use the resumeEvoTest function instead, which should accept a session ID directly
      const response = await evoTesterApi.resumeEvoTest(activeSessionId);
      
      if (response.success) {
        showSuccessToast('Evolution session resumed');
        queryClient.invalidateQueries(['evoTester', 'status', activeSessionId]);
      } else {
        showErrorToast('Failed to resume evolution session');
      }
    } catch (err) {
      console.error('Error resuming session:', err);
      showErrorToast('Error resuming evolution session');
    }
  };

  // Handle selecting a session from the list
  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
  };

  // Handle promoting a strategy to production use
  const handleSaveStrategy = async (strategyToSave: EvoStrategy) => {
    try {
      const response = await evoTesterApi.promoteStrategy(strategyToSave);

      if (response.success) {
        showSuccessToast('Strategy saved successfully');
      } else {
        showErrorToast('Failed to save strategy');
      }
    } catch (err) {
      console.error('Error saving strategy:', err);
      showErrorToast('Error saving strategy');
    }
  };

  // Handle adding research discoveries to evolution candidates
  const handleAddToEvolution = async (symbols: string[], researchConfig?: any) => {
    try {
      // Start a new evolution session with the selected symbols
      const config = researchConfig || {
        population_size: 100,
        generations: 50,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        target_asset: symbols[0], // Use first symbol as primary target
        symbols: symbols, // Include all selected diamonds
        optimization_metric: 'sharpe',
        sentiment_weight: 0.3,
        news_impact_weight: 0.2,
        intelligence_snowball: true
      };

      await handleStartNewSession(config);
      showSuccessToast(`Evolution started with ${symbols.length} research candidates!`);
    } catch (err) {
      console.error('Error starting evolution with research discoveries:', err);
      showErrorToast('Failed to start evolution with selected research discoveries');
    }
  };

  // Format the progress data for the fitness trend chart
  const formatChartData = () => {
    if (!generations || generations.length === 0) return [];
    
    return generations.map(gen => ({
      timestamp: gen.generation.toString(), // Add timestamp property required by TimeSeriesChart
      generation: gen.generation,
      bestFitness: gen.bestFitness,
      avgFitness: gen.averageFitness,
      diversityScore: gen.diversityScore || 0,
    }));
  };

  return (
    <div className={`${className}`}>
      {/* [1] Evolution Status Bar - Top status overview and session management */}
      <EvolutionStatusBar
        activeSessions={activeSessions?.length || 0}
        totalStrategies={activeSessions?.reduce((acc, session) => acc + (session.totalStrategies || 100), 0) || 0}
        bestFitness={Math.max(...(activeSessions?.map(s => s.bestFitness || 0) || [0]))}
        marketRegime="Bull Market"
        lastDeployment={new Date(Date.now() - 2 * 60 * 60 * 1000)}
        activeSymbols={['SPY', 'QQQ', 'AAPL']}
        sentimentScore={0.67}
        newsImpactScore={0.45}
      />

      {/* Vertical Card Layout - Clear Separation */}
      <div className="space-y-8 mt-6">
        {/* [3] Evolution Progress Card */}
        <Card className="min-h-80">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Evolution Progress</span>
              <div className="flex space-x-2">
                {progress?.status === 'running' ? (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handlePauseSession}
                      disabled={!activeSessionId || progress?.status !== 'running'}
                    >
                      <Pause className="h-4 w-4 mr-1" /> Pause
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={handleStopSession}
                      disabled={!activeSessionId || !['running', 'paused'].includes(progress?.status || '')}
                    >
                      <StopCircle className="h-4 w-4 mr-1" /> Stop
                    </Button>
                  </>
                ) : progress?.status === 'paused' ? (
                  <>
                    <Button
                      size="sm"
                      variant="default"
                      onClick={handleResumeSession}
                      disabled={!activeSessionId || progress?.status !== 'paused'}
                    >
                      <Play className="h-4 w-4 mr-1" /> Resume
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={handleStopSession}
                      disabled={!activeSessionId || !['running', 'paused'].includes(progress?.status || '')}
                    >
                      <StopCircle className="h-4 w-4 mr-1" /> Stop
                    </Button>
                  </>
                ) : (
                  <Button
                    size="sm"
                    variant="default"
                    onClick={() => {
                      // Show configuration modal or start with defaults
                      handleStartNewSession({
                        population_size: 100,
                        generations: 50,
                        mutation_rate: 0.1,
                        crossover_rate: 0.8,
                        target_asset: 'SPY', // Default to S&P 500
                        optimization_metric: 'sharpe',
                        symbols: 'all', // Use all available symbols by default
                        sentiment_weight: 0.3, // Include sentiment in fitness
                        news_impact_weight: 0.2, // Include news impact in fitness
                        intelligence_snowball: true // Enable intelligence accumulation
                      });
                    }}
                    disabled={isRunning}
                  >
                    <Play className="h-4 w-4 mr-1" /> Start New
                  </Button>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {activeSessionId && progress ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm text-foreground">Session ID</div>
                    <div className="text-md font-medium">{activeSessionId.substring(0, 8)}...</div>
                  </div>
                  <div>
                    <div className="text-sm text-foreground">Status</div>
                    <StatusBadge
                      variant={progress?.status === 'running' ? 'bull' :
                               progress?.status === 'paused' ? 'warning' :
                               progress?.status === 'completed' ? 'info' :
                               'bear'}
                      withDot
                      pulse={progress?.status === 'running'}
                    >
                      {progress?.status ? progress.status.charAt(0).toUpperCase() + progress.status.slice(1) : 'Unknown'}
                    </StatusBadge>
                  </div>
                  <div>
                    <div className="text-sm text-foreground">Generation</div>
                    <div className="text-md font-medium">
                      {progress?.currentGeneration || 0} / {progress?.totalGenerations || 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-foreground">Best Fitness</div>
                    <div className="text-md font-medium text-green-600">
                      {progress?.bestFitness?.toFixed(4) || '0.0000'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-foreground">Avg Fitness</div>
                    <div className="text-md font-medium">
                      {progress?.averageFitness?.toFixed(4) || '0.0000'}
                    </div>
                  </div>
                </div>

                <div className="relative pt-4">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`${styles.dashboardProgressBar} ${styles[`progress${Math.round(Math.min(Math.max((progress?.progress || 0), 0) * 100, 100) / 5) * 5}`]}`}
                    ></div>
                  </div>
                  <div className="mt-2 flex justify-between text-xs text-foreground">
                    <span>0%</span>
                    <span>Progress: {((progress?.progress || 0) * 100).toFixed(1)}%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div className="pt-4">
                  <Tabs defaultValue="fitness-trends">
                    <TabsList className="mb-4">
                      <TabsTrigger value="fitness-trends">Fitness Trends</TabsTrigger>
                      <TabsTrigger value="diversity">Population Diversity</TabsTrigger>
                      <TabsTrigger value="strategy-distribution">Strategy Distribution</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="fitness-trends" className="h-64">
                      <FitnessTrendChart data={formatChartData()} />
                    </TabsContent>
                    
                    <TabsContent value="diversity" className="h-64">
                      <TimeSeriesChart 
                        data={formatChartData()}
                        series={[
                          {
                            name: 'Population Diversity',
                            dataKey: 'diversityScore',
                            color: '#10B981' // green
                          }
                        ]}
                        xAxisDataKey="generation"
                        showLegend={false}
                      />
                    </TabsContent>
                    
                    <TabsContent value="strategy-distribution" className="h-64">
                      <div className="flex items-center justify-center h-full text-foreground">
                        Strategy distribution visualization will be available after a few generations
                      </div>
                    </TabsContent>
                  </Tabs>
                </div>

                {error && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
                    <div className="flex items-center">
                      <AlertTriangle className="h-4 w-4 mr-2" />
                      <span>Error: {error}</span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-foreground">
                <div className="text-xl mb-2">No active evolution session</div>
                <div className="text-sm mb-4">Start a new session or select an existing one</div>
                <Button 
                  onClick={() => {
                    // Show configuration modal or start with defaults
                    handleStartNewSession({
                      population_size: 100,
                      generations: 50,
                      mutation_rate: 0.1,
                      crossover_rate: 0.8,
                      target_asset: 'BTC-USD',
                      optimization_metric: 'sharpe'
                    });
                  }}
                >
                  <Play className="h-4 w-4 mr-1" /> Start New Evolution
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* [4] Active Sessions & History */}
        <Card className="min-h-64">
          <CardHeader>
            <CardTitle>
              <Tabs
                defaultValue="active"
                onValueChange={(value) => setViewMode(value as 'active' | 'history')}
              >
                <TabsList className="w-full h-8">
                  <TabsTrigger value="active" className="text-xs flex-1">[4] Active Sessions</TabsTrigger>
                  <TabsTrigger value="history" className="text-xs flex-1">[5] Session History</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {viewMode === 'active' ? (
              <ActiveSessionsList
                sessions={activeSessions || []}
                onSelectSession={handleSelectSession}
                activeSessionId={activeSessionId || undefined}
                isLoading={sessionsLoading}
              />
            ) : (
              <div className="space-y-3">
                {historyLoading ? (
                  <div className="flex items-center justify-center h-40">
                    <RefreshCw className="h-5 w-5 animate-spin text-foreground" />
                  </div>
                ) : (sessionHistory && sessionHistory.length > 0) ? (
                  sessionHistory.map((session) => (
                    <div 
                      key={session.id}
                      className="p-3 border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer"
                      onClick={() => handleSelectSession(session.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">Session {session.id.substring(0, 8)}</div>
                          <div className="text-xs text-foreground">{new Date(session.date).toLocaleString()}</div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-blue-600">
                            {session.bestFitness.toFixed(4)}
                          </span>
                          <ArrowUpRight className="h-4 w-4 text-foreground" />
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="flex items-center justify-center h-40 text-foreground">
                    No past sessions found
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* [5] Research & Discovery Hub - News analysis, fundamental research, strategy hypotheses, market discovery */}
        <div className="min-h-72">
          <ResearchDiscoveryHub
            onStartEvolutionWithSymbols={handleAddToEvolution}
          />
        </div>

        {/* [6] Evolution Results Hub - Strategy results, explanation, pipeline, deployment */}
        <div className="min-h-80">
          <EvolutionResultsHub
            topStrategies={result?.topStrategies || []}
            onSelectStrategy={setSelectedStrategy}
            onSaveStrategy={handleSaveStrategy}
          />
        </div>

        {/* [7] Evolution Lifecycle View - Timeline, champion lineage, population dynamics */}
        <div className="min-h-64">
          <EvoLifecycleView sessionId={activeSessionId || undefined} />
        </div>

        {/* [8] Evolution Sandbox - Auto-triggers, capital management, automated experiments */}
        <div className="min-h-64">
          <EvolutionSandbox />
        </div>

        {/* [9] Promotion Pipeline - Strategy promotion criteria, validation gates */}
        <div className="min-h-64">
          <PromotionPipeline />
        </div>
      </div>

      {/* Strategy Parameters Modal would go here */}
      {selectedStrategy && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card rounded-lg w-full max-w-3xl max-h-[90vh] overflow-auto border border-border text-foreground">
            <div className="p-4 border-b border-border flex justify-between items-center">
              <h2 className="text-xl font-semibold">
                Strategy Details: {selectedStrategy.name || `Strategy ${selectedStrategy.id?.substring(0, 8) || 'Unknown'}`}
              </h2>
              <button 
                onClick={() => setSelectedStrategy(null)}
                className="text-foreground hover:text-foreground/80"
              >
                &times;
              </button>
            </div>
            <div className="p-6">
              <StrategyParametersView strategy={selectedStrategy} />
            </div>
            <div className="p-4 border-t border-border flex justify-end">
              <Button 
                variant="outline" 
                className="mr-2"
                onClick={() => setSelectedStrategy(null)}
              >
                Close
              </Button>
              <Button onClick={() => handleSaveStrategy(selectedStrategy)}>
                <Save className="h-4 w-4 mr-1" /> Save Strategy
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvoTesterDashboard;
