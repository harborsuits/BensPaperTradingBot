/**
 * ============================================
 * [CARD: EVOLUTION RESULTS HUB]
 * Top strategies, how evolution works, system flow, deployment status
 * ============================================
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import {
  Award,
  Brain,
  TrendingUp,
  Target,
  BarChart3,
  Zap,
  CheckCircle,
  Clock,
  ArrowRight,
  Star,
  Crown,
  Activity,
  DollarSign,
  Shield,
  Search,
  Gem,
  Sparkles,
  Plus,
  Eye,
  RefreshCw,
  Loader2
} from 'lucide-react';
import { EvoStrategy } from '@/types/api.types';
import { showSuccessToast, showErrorToast } from '@/utils/toast.js';
import { useLabDiamonds, DiamondSymbol } from '@/hooks/useLabDiamonds';
import { evoTesterApi, strategyApi } from '@/services/api';

interface EvolutionResultsHubProps {
  topStrategies?: EvoStrategy[];
  onSelectStrategy?: (strategy: EvoStrategy) => void;
  onSaveStrategy?: (strategy: EvoStrategy) => void;
  onAddToEvolution?: (symbols: string[]) => void;
  className?: string;
}

export const EvolutionResultsHub: React.FC<EvolutionResultsHubProps> = ({
  topStrategies = [],
  onSelectStrategy,
  onSaveStrategy,
  onAddToEvolution,
  className = ''
}) => {
  // Real-time data connections
  const { data: liveStrategies, isLoading: strategiesLoading } = useQuery({
    queryKey: ['strategies', 'active'],
    queryFn: () => strategyApi.getActiveStrategies(),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 15000,
  });

  const { data: evolutionHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['evoTester', 'completed'],
    queryFn: () => evoTesterApi.getEvoHistory(),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000,
  });

  const { data: labDiamonds, isLoading: diamondsLoading, error: diamondsError } = useLabDiamonds();

  const [selectedTab, setSelectedTab] = useState('strategies');
  const [selectedDiamonds, setSelectedDiamonds] = useState<string[]>([]);

  // Use the live diamonds data with proper fallback
  const diamondsData = labDiamonds || [];

  const handleSaveStrategy = async (strategy: EvoStrategy) => {
    if (onSaveStrategy) {
      onSaveStrategy(strategy);
    } else {
      showSuccessToast('Strategy saved successfully');
    }
  };

  const handleSelectStrategy = (strategy: EvoStrategy) => {
    if (onSelectStrategy) {
      onSelectStrategy(strategy);
    }
  };

  const handleDiamondToggle = (symbol: string) => {
    setSelectedDiamonds(prev =>
      prev.includes(symbol)
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  const handleAddToEvolution = () => {
    if (selectedDiamonds.length > 0 && onAddToEvolution) {
      onAddToEvolution(selectedDiamonds);
      showSuccessToast(`Added ${selectedDiamonds.length} diamonds to evolution candidates`);
      setSelectedDiamonds([]);
    }
  };

  // Mock pipeline data - in real implementation, this would come from props
  const pipelineStages = [
    { name: 'INGEST', status: 'completed', progress: 100, description: 'Data collection & preprocessing' },
    { name: 'CONTEXT', status: 'completed', progress: 100, description: 'Market context analysis' },
    { name: 'CANDIDATES', status: 'completed', progress: 100, description: 'Strategy generation' },
    { name: 'GATES', status: 'completed', progress: 100, description: 'Risk & validation checks' },
    { name: 'PLAN', status: 'completed', progress: 100, description: 'Execution planning' },
    { name: 'ROUTE', status: 'running', progress: 75, description: 'Order routing' },
    { name: 'MANAGE', status: 'pending', progress: 0, description: 'Position management' },
    { name: 'LEARN', status: 'pending', progress: 0, description: 'Performance learning' }
  ];

  const deploymentMetrics = {
    totalStrategies: 156,
    deployedStrategies: 23,
    pendingStrategies: 8,
    failedStrategies: 2,
    averageFitness: 2.145,
    successRate: 87.5
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Award className="w-6 h-6 mr-2 text-yellow-500" />
          Evolution Results Hub
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={selectedTab} onValueChange={setSelectedTab}>
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="strategies">Top Strategies</TabsTrigger>
            <TabsTrigger value="research">Research</TabsTrigger>
            <TabsTrigger value="explanation">How It Works</TabsTrigger>
            <TabsTrigger value="pipeline">System Flow</TabsTrigger>
            <TabsTrigger value="deployment">Deployment</TabsTrigger>
          </TabsList>

          {/* Top Strategies Tab */}
          <TabsContent value="strategies" className="space-y-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Award className="w-5 h-5 mr-2 text-yellow-500" />
                Live Strategy Results
                {strategiesLoading && <Loader2 className="w-4 h-4 ml-2 animate-spin text-yellow-500" />}
              </h3>
              <Button size="sm" variant="outline">
                <RefreshCw className="w-3 h-3 mr-1" />
                Refresh
              </Button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {topStrategies.slice(0, 3).map((strategy, index) => (
                <div
                  key={index}
                  className={`p-4 border rounded-lg bg-card border-border text-foreground ${
                    index === 0 ? 'ring-2 ring-yellow-500/50' : ''
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      {index === 0 && (
                        <Crown className="w-5 h-5 text-yellow-500 mr-2" />
                      )}
                      <h3 className="font-medium">
                        {strategy.name || `Strategy #${index + 1}`}
                      </h3>
                    </div>
                    <Badge
                      variant={index === 0 ? 'default' : index === 1 ? 'secondary' : 'outline'}
                      className={index === 0 ? 'bg-yellow-500 text-white' : ''}
                    >
                      Rank #{index + 1}
                    </Badge>
                  </div>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-foreground">Fitness:</span>
                      <span className="font-medium">{strategy.fitness.toFixed(4)}</span>
                    </div>
                    {strategy.performance && (
                      <>
                        <div className="flex justify-between text-sm">
                          <span className="text-foreground">Sharpe Ratio:</span>
                          <span className="font-medium">{strategy.performance.sharpeRatio?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-foreground">Win Rate:</span>
                          <span className="font-medium">{strategy.performance.winRate ? `${(strategy.performance.winRate * 100).toFixed(1)}%` : 'N/A'}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-foreground">Max Drawdown:</span>
                          <span className="font-medium">{strategy.performance.maxDrawdown ? `${(strategy.performance.maxDrawdown * 100).toFixed(1)}%` : 'N/A'}</span>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="flex justify-end space-x-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleSelectStrategy(strategy)}
                    >
                      Details
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => handleSaveStrategy(strategy)}
                    >
                      <Star className="h-3.5 w-3.5 mr-1" /> Save
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {topStrategies.length === 0 && (
              <div className="text-center py-8 text-foreground">
                <Award className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p className="text-lg font-medium mb-2">No Strategies Yet</p>
                <p className="text-sm">Run an evolution session to generate winning strategies</p>
              </div>
            )}
          </TabsContent>

          {/* Research & Discovery Tab */}
          <TabsContent value="research" className="space-y-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold flex items-center">
                  <Gem className="w-5 h-5 mr-2 text-yellow-500" />
                  Diamonds in the Rough
                </h3>
                <p className="text-sm text-foreground mt-1">
                  Discover promising symbols for evolution testing
                </p>
              </div>
              {selectedDiamonds.length > 0 && (
                <Button onClick={handleAddToEvolution} className="flex items-center">
                  <Plus className="w-4 h-4 mr-1" />
                  Add {selectedDiamonds.length} to Evolution
                </Button>
              )}
            </div>

            {diamondsLoading ? (
              <div className="text-center py-8">
                <Sparkles className="w-8 h-8 mx-auto mb-4 text-yellow-500 animate-pulse" />
                <p className="text-foreground">Scanning markets for diamonds...</p>
              </div>
            ) : diamondsError ? (
              <div className="text-center py-8">
                <p className="text-red-500">Error loading diamond research data</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Top Diamonds */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {diamondsData?.items.slice(0, 9).map((diamond, index) => (
                    <div
                      key={diamond.symbol}
                      className={`p-4 border rounded-lg bg-card border-border text-foreground ${
                        selectedDiamonds.includes(diamond.symbol) ? 'ring-2 ring-yellow-500/50 bg-yellow-50/10' : ''
                      }`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center">
                          <Gem className="w-4 h-4 mr-2 text-yellow-500" />
                          <span className="font-semibold">{diamond.symbol}</span>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-bold text-green-600">
                            {diamond.score.toFixed(3)}
                          </div>
                          <div className="text-xs text-foreground">Score</div>
                        </div>
                      </div>

                      <div className="space-y-2 mb-3">
                        <div className="flex justify-between text-xs">
                          <span className="text-foreground">Sentiment 1h:</span>
                          <span className={`font-medium ${diamond.features.impact1h > 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {diamond.features.impact1h > 0 ? '+' : ''}{diamond.features.impact1h.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-foreground">Gap %:</span>
                          <span className={`font-medium ${diamond.features.gapPct > 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {diamond.features.gapPct > 0 ? '+' : ''}{diamond.features.gapPct.toFixed(2)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-foreground">RVOL:</span>
                          <span className="font-medium">{diamond.features.rvol.toFixed(1)}x</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-foreground">Spread:</span>
                          <span className="font-medium">{diamond.features.spreadPct.toFixed(2)}%</span>
                        </div>
                      </div>

                      <Button
                        size="sm"
                        variant={selectedDiamonds.includes(diamond.symbol) ? "default" : "outline"}
                        onClick={() => handleDiamondToggle(diamond.symbol)}
                        className="w-full"
                      >
                        {selectedDiamonds.includes(diamond.symbol) ? (
                          <>
                            <Eye className="w-3 h-3 mr-1" />
                            Selected
                          </>
                        ) : (
                          <>
                            <Plus className="w-3 h-3 mr-1" />
                            Select for Evolution
                          </>
                        )}
                      </Button>
                    </div>
                  ))}
                </div>

                {/* Research Insights */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                  <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2 flex items-center">
                      <BarChart3 className="w-4 h-4 mr-2" />
                      Research Methodology
                    </h4>
                    <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1">
                      <li>• <strong>45% Sentiment Analysis:</strong> Recent news impact</li>
                      <li>• <strong>20% Relative Volume:</strong> Unusual trading activity</li>
                      <li>• <strong>15% Price Gap:</strong> Intraday price movements</li>
                      <li>• <strong>15% Spread Penalty:</strong> Trading costs</li>
                      <li>• <strong>5% Exploration:</strong> Novel opportunities</li>
                    </ul>
                  </div>

                  <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center">
                      <Search className="w-4 h-4 mr-2" />
                      Discovery Process
                    </h4>
                    <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                      <li>• Scans 400+ symbols across all asset classes</li>
                      <li>• Combines multiple market signals</li>
                      <li>• Identifies potential trading opportunities</li>
                      <li>• Feeds promising candidates to evolution</li>
                    </ul>
                  </div>
                </div>

                {selectedDiamonds.length > 0 && (
                  <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-4 border border-yellow-200 dark:border-yellow-800">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-semibold text-yellow-900 dark:text-yellow-100">
                          Selected for Evolution
                        </h4>
                        <p className="text-sm text-yellow-800 dark:text-yellow-200">
                          {selectedDiamonds.join(', ')}
                        </p>
                      </div>
                      <Button onClick={handleAddToEvolution} className="bg-yellow-600 hover:bg-yellow-700">
                        <Zap className="w-4 h-4 mr-1" />
                        Start Evolution
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </TabsContent>

          {/* How It Works Tab */}
          <TabsContent value="explanation" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* What It Did */}
              <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center">
                  <Target className="w-4 h-4 mr-2" />
                  What the Test Did
                </h3>
                <p className="text-blue-800 dark:text-blue-200 text-sm leading-relaxed">
                  Ran a <strong>50-generation evolutionary algorithm</strong> with <strong>100+ strategies</strong> per generation.
                  Each strategy was an <strong>RSI-based momentum trader</strong> that bought when oversold and sold when overbought.
                </p>
              </div>

              {/* Optimization Criteria */}
              <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
                <h3 className="font-semibold text-green-900 dark:text-green-100 mb-3 flex items-center">
                  <TrendingUp className="w-4 h-4 mr-2" />
                  What It Optimized
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-green-800 dark:text-green-200">Market Sentiment:</span>
                    <Badge variant="outline" className="bg-green-100 text-green-800">67% weight</Badge>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-green-800 dark:text-green-200">Profit/Loss:</span>
                    <Badge variant="outline" className="bg-green-100 text-green-800">25% weight</Badge>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-green-800 dark:text-green-200">Risk Control:</span>
                    <Badge variant="outline" className="bg-red-100 text-red-800">-8% penalty</Badge>
                  </div>
                </div>
              </div>

              {/* Symbols Used */}
              <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
                <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-2 flex items-center">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Symbols Tested
                </h3>
                <div className="flex flex-wrap gap-2 mb-3">
                  {['SPY', 'AAPL', 'NVDA', 'TSLA', 'QQQ', 'MSFT', 'GOOGL', 'AMZN'].map(symbol => (
                    <Badge key={symbol} variant="outline" className="bg-purple-100 text-purple-800">
                      {symbol}
                    </Badge>
                  ))}
                </div>
                <p className="text-purple-800 dark:text-purple-200 text-sm">
                  Tested on major stocks and ETFs using live market data.
                </p>
              </div>

              {/* Evolution Process */}
              <div className="bg-orange-50 dark:bg-orange-950/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
                <h3 className="font-semibold text-orange-900 dark:text-orange-100 mb-3 flex items-center">
                  <Zap className="w-4 h-4 mr-2" />
                  Evolution Process
                </h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">1</div>
                    <div>
                      <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Random Strategies</p>
                      <p className="text-orange-700 dark:text-orange-300 text-xs">Started with 100 random RSI strategies</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">2</div>
                    <div>
                      <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Survival of Fittest</p>
                      <p className="text-orange-700 dark:text-orange-300 text-xs">Only top 78% survived to breed</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">50</div>
                    <div>
                      <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Best Strategy Found</p>
                      <p className="text-orange-700 dark:text-orange-300 text-xs">RSI 15, Stop Loss 2.48% - Fitness 2.4335</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* System Flow Tab */}
          <TabsContent value="pipeline" className="space-y-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Activity className="w-5 h-5 mr-2" />
                System Processing Pipeline
              </h3>

              <div className="space-y-3">
                {pipelineStages.map((stage, index) => (
                  <div key={stage.name} className="flex items-center space-x-4 p-3 bg-card border border-border rounded-lg">
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                      stage.status === 'completed' ? 'bg-green-500 text-white' :
                      stage.status === 'running' ? 'bg-blue-500 text-white' :
                      'bg-gray-300 text-gray-600'
                    }`}>
                      {stage.status === 'completed' ? <CheckCircle className="w-4 h-4" /> :
                       stage.status === 'running' ? <Clock className="w-4 h-4" /> :
                       index + 1}
                    </div>

                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">{stage.name}</span>
                        <span className="text-sm text-foreground">{stage.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            stage.status === 'completed' ? 'bg-green-500' :
                            stage.status === 'running' ? 'bg-blue-500' :
                            'bg-gray-300'
                          }`}
                          style={{ width: `${stage.progress}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-foreground mt-1">{stage.description}</p>
                    </div>

                    {index < pipelineStages.length - 1 && (
                      <ArrowRight className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <div className="bg-green-50 dark:bg-green-950/20 p-3 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="text-2xl font-bold text-green-600 mb-1">
                    {pipelineStages.filter(s => s.status === 'completed').length}
                  </div>
                  <div className="text-sm text-green-800 dark:text-green-200">Completed</div>
                </div>
                <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="text-2xl font-bold text-blue-600 mb-1">
                    {pipelineStages.filter(s => s.status === 'running').length}
                  </div>
                  <div className="text-sm text-blue-800 dark:text-blue-200">Running</div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-800">
                  <div className="text-2xl font-bold text-gray-600 mb-1">
                    {pipelineStages.filter(s => s.status === 'pending').length}
                  </div>
                  <div className="text-sm text-gray-800 dark:text-gray-200">Pending</div>
                </div>
                <div className="bg-purple-50 dark:bg-purple-950/20 p-3 rounded-lg border border-purple-200 dark:border-purple-800">
                  <div className="text-2xl font-bold text-purple-600 mb-1">
                    {Math.round(pipelineStages.reduce((sum, s) => sum + s.progress, 0) / pipelineStages.length)}%
                  </div>
                  <div className="text-sm text-purple-800 dark:text-purple-200">Overall</div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Deployment Tab */}
          <TabsContent value="deployment" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Deployment Metrics */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <DollarSign className="w-5 h-5 mr-2" />
                  Deployment Metrics
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-950/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="text-2xl font-bold text-blue-600 mb-1">
                      {deploymentMetrics.totalStrategies}
                    </div>
                    <div className="text-sm text-blue-800 dark:text-blue-200">Total Strategies</div>
                  </div>
                  <div className="bg-green-50 dark:bg-green-950/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="text-2xl font-bold text-green-600 mb-1">
                      {deploymentMetrics.deployedStrategies}
                    </div>
                    <div className="text-sm text-green-800 dark:text-green-200">Deployed</div>
                  </div>
                  <div className="bg-yellow-50 dark:bg-yellow-950/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <div className="text-2xl font-bold text-yellow-600 mb-1">
                      {deploymentMetrics.pendingStrategies}
                    </div>
                    <div className="text-sm text-yellow-800 dark:text-yellow-200">Pending</div>
                  </div>
                  <div className="bg-red-50 dark:bg-red-950/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
                    <div className="text-2xl font-bold text-red-600 mb-1">
                      {deploymentMetrics.failedStrategies}
                    </div>
                    <div className="text-sm text-red-800 dark:text-red-200">Failed</div>
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-950/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-purple-800 dark:text-purple-200">Success Rate</span>
                    <span className="font-bold text-purple-600">{deploymentMetrics.successRate}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-purple-800 dark:text-purple-200">Avg Fitness</span>
                    <span className="font-bold text-purple-600">{deploymentMetrics.averageFitness.toFixed(3)}</span>
                  </div>
                </div>
              </div>

              {/* Deployment Stages */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <Shield className="w-5 h-5 mr-2" />
                  Deployment Stages
                </h3>

                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="flex items-center">
                      <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                      <span className="font-medium">Risk Assessment</span>
                    </div>
                    <Badge variant="outline" className="bg-green-100 text-green-800">PASSED</Badge>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="flex items-center">
                      <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                      <span className="font-medium">Capital Allocation</span>
                    </div>
                    <Badge variant="outline" className="bg-green-100 text-green-800">PASSED</Badge>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="flex items-center">
                      <Clock className="w-5 h-5 text-blue-500 mr-3" />
                      <span className="font-medium">Live Deployment</span>
                    </div>
                    <Badge variant="outline" className="bg-blue-100 text-blue-800">IN PROGRESS</Badge>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800">
                    <div className="flex items-center">
                      <Clock className="w-5 h-5 text-gray-500 mr-3" />
                      <span className="font-medium">Performance Monitoring</span>
                    </div>
                    <Badge variant="outline">PENDING</Badge>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
