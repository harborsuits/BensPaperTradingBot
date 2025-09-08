import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import {
  BarChart3,
  TrendingUp,
  Target,
  Zap,
  Dna,
  PieChart,
  Activity,
  Settings,
  Eye,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

interface ParameterImportance {
  name: string;
  importance: number;
  impact: 'high' | 'medium' | 'low';
  optimalRange: string;
  frequency: number;
}

interface MarketConditionPreference {
  condition: string;
  performance: number;
  frequency: number;
  strategies: number;
}

interface RiskReturnProfile {
  strategyId: string;
  name: string;
  sharpeRatio: number;
  sortinoRatio: number;
  winRate: number;
  maxDrawdown: number;
  profitFactor: number;
  expectancy: number;
}

interface StrategyDNA {
  strategyId: string;
  name: string;
  generation: number;
  parentIds: string[];
  parameters: Record<string, any>;
  fitness: number;
  creationMethod: 'mutation' | 'crossover' | 'random';
  survivalTime: number;
  offspringCount: number;
}

interface StrategyIntelligenceProps {
  className?: string;
}

const StrategyIntelligence: React.FC<StrategyIntelligenceProps> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState('parameters');
  const [expandedStrategy, setExpandedStrategy] = useState<string | null>(null);

  const parameterImportance: ParameterImportance[] = [
    {
      name: 'RSI Period',
      importance: 0.85,
      impact: 'high',
      optimalRange: '12-16',
      frequency: 78
    },
    {
      name: 'Stop Loss %',
      importance: 0.72,
      impact: 'high',
      optimalRange: '1.8-2.3%',
      frequency: 65
    },
    {
      name: 'Take Profit %',
      importance: 0.68,
      impact: 'high',
      optimalRange: '3.5-4.2%',
      frequency: 59
    },
    {
      name: 'SMA Fast Period',
      importance: 0.55,
      impact: 'medium',
      optimalRange: '8-12',
      frequency: 42
    },
    {
      name: 'SMA Slow Period',
      importance: 0.43,
      impact: 'medium',
      optimalRange: '18-25',
      frequency: 38
    }
  ];

  const marketConditions: MarketConditionPreference[] = [
    {
      condition: 'Bull Trending (SPY +15%)',
      performance: 2.34,
      frequency: 45,
      strategies: 89
    },
    {
      condition: 'High Volatility (VIX > 25)',
      performance: 1.87,
      frequency: 32,
      strategies: 67
    },
    {
      condition: 'Bear Trending (SPY -10%)',
      performance: 1.23,
      frequency: 18,
      strategies: 34
    },
    {
      condition: 'Positive Sentiment (>70%)',
      performance: 2.45,
      frequency: 28,
      strategies: 52
    },
    {
      condition: 'Negative News Impact',
      performance: 1.67,
      frequency: 22,
      strategies: 41
    },
    {
      condition: 'Low Volatility (VIX < 15)',
      performance: 0.95,
      frequency: 15,
      strategies: 28
    }
  ];

  const riskReturnProfiles: RiskReturnProfile[] = [
    {
      strategyId: 'evo-247',
      name: 'RSI-Momentum-V2',
      sharpeRatio: 2.34,
      sortinoRatio: 1.87,
      winRate: 0.67,
      maxDrawdown: 0.12,
      profitFactor: 2.1,
      expectancy: 0.023
    },
    {
      strategyId: 'evo-189',
      name: 'Volatility-Breakout',
      sharpeRatio: 1.98,
      sortinoRatio: 1.65,
      winRate: 0.58,
      maxDrawdown: 0.15,
      profitFactor: 1.8,
      expectancy: 0.018
    },
    {
      strategyId: 'evo-156',
      name: 'MeanReversion-Pro',
      sharpeRatio: 1.76,
      sortinoRatio: 1.42,
      winRate: 0.71,
      maxDrawdown: 0.08,
      profitFactor: 2.3,
      expectancy: 0.015
    }
  ];

  const strategyDNA: StrategyDNA[] = [
    {
      strategyId: 'evo-247',
      name: 'RSI-Momentum-V2',
      generation: 47,
      parentIds: ['evo-201', 'evo-189'],
      parameters: {
        rsiPeriod: 14,
        overboughtLevel: 68,
        oversoldLevel: 32,
        stopLossPct: 2.1,
        takeProfitPct: 3.8
      },
      fitness: 2.34,
      creationMethod: 'crossover',
      survivalTime: 12,
      offspringCount: 3
    },
    {
      strategyId: 'evo-189',
      name: 'Volatility-Breakout',
      generation: 32,
      parentIds: ['evo-145', 'evo-167'],
      parameters: {
        volatilityThreshold: 0.25,
        breakoutStrength: 1.8,
        confirmationPeriod: 3,
        stopLossPct: 1.5,
        takeProfitPct: 4.2
      },
      fitness: 1.98,
      creationMethod: 'mutation',
      survivalTime: 27,
      offspringCount: 8
    }
  ];

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getCreationMethodColor = (method: string) => {
    switch (method) {
      case 'crossover':
        return 'bg-blue-100 text-blue-800';
      case 'mutation':
        return 'bg-purple-100 text-purple-800';
      case 'random':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Card className={className}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Strategy Intelligence</h3>
            <p className="text-sm text-gray-600">Understanding what makes strategies successful</p>
          </div>
          <Badge variant="outline" className="bg-purple-50 border-purple-300 text-purple-700">
            <Dna className="w-3 h-3 mr-1" />
            AI Analysis
          </Badge>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full">
            <TabsTrigger value="parameters" className="flex-1">
              <Settings className="w-4 h-4 mr-2" />
              Parameters
            </TabsTrigger>
            <TabsTrigger value="markets" className="flex-1">
              <BarChart3 className="w-4 h-4 mr-2" />
              Markets
            </TabsTrigger>
            <TabsTrigger value="risk" className="flex-1">
              <TrendingUp className="w-4 h-4 mr-2" />
              Risk Profile
            </TabsTrigger>
            <TabsTrigger value="dna" className="flex-1">
              <Dna className="w-4 h-4 mr-2" />
              Strategy DNA
            </TabsTrigger>
            <TabsTrigger value="intelligence" className="flex-1">
              <Brain className="w-4 h-4 mr-2" />
              Intelligence
            </TabsTrigger>
          </TabsList>

          {/* Parameter Importance */}
          <TabsContent value="parameters" className="mt-6">
            <div className="space-y-4">
              <h4 className="font-medium flex items-center">
                <Target className="w-4 h-4 mr-2 text-blue-600" />
                Parameter Importance Analysis
              </h4>
              <div className="space-y-3">
                {parameterImportance.map((param, index) => (
                  <div key={param.name} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <span className="font-medium">{param.name}</span>
                        <Badge variant="outline" className={getImpactColor(param.impact)}>
                          {param.impact} impact
                        </Badge>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-blue-600">
                          {(param.importance * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-gray-500">importance</div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm text-gray-600">
                      <span>Optimal Range: {param.optimalRange}</span>
                      <span>Used in {param.frequency}% of strategies</span>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${param.importance * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Market Condition Preferences */}
          <TabsContent value="markets" className="mt-6">
            <div className="space-y-6">
              {/* Symbol Evolution Status */}
              <div>
                <h4 className="font-medium flex items-center mb-4">
                  <Activity className="w-4 h-4 mr-2 text-blue-600" />
                  Symbol Evolution Intelligence
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'BTC-USD'].map((symbol) => (
                    <div key={symbol} className="border rounded-lg p-3 bg-gray-50">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm">{symbol}</span>
                        <Badge variant="outline" className="text-xs bg-green-50 text-green-700">
                          Active
                        </Badge>
                      </div>
                      <div className="space-y-1 text-xs text-gray-600">
                        <div>Evolved: 47 strategies</div>
                        <div>Best Fitness: 2.34</div>
                        <div>Sentiment: +67%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Market Condition Performance */}
              <div>
                <h4 className="font-medium flex items-center">
                  <PieChart className="w-4 h-4 mr-2 text-green-600" />
                  Market Condition Performance
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                {marketConditions.map((condition, index) => (
                  <div key={condition.condition} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{condition.condition}</span>
                      <Badge variant="outline" className="bg-green-50 border-green-300 text-green-700">
                        {condition.performance.toFixed(2)} Sharpe
                      </Badge>
                    </div>
                    <div className="space-y-1 text-sm text-gray-600">
                      <div>Performance: {condition.performance.toFixed(2)} Sharpe</div>
                      <div>Frequency: {condition.frequency}% of time</div>
                      <div>Strategies: {condition.strategies} adapted</div>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full"
                        style={{ width: `${(condition.performance / 2.5) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Risk-Return Profiles */}
          <TabsContent value="risk" className="mt-6">
            <div className="space-y-4">
              <h4 className="font-medium flex items-center">
                <Activity className="w-4 h-4 mr-2 text-orange-600" />
                Risk-Return Profile Analysis
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Strategy</th>
                      <th className="text-center py-2">Sharpe</th>
                      <th className="text-center py-2">Sortino</th>
                      <th className="text-center py-2">Win Rate</th>
                      <th className="text-center py-2">Max DD</th>
                      <th className="text-center py-2">Profit Factor</th>
                      <th className="text-center py-2">Expectancy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {riskReturnProfiles.map((profile, index) => (
                      <tr key={profile.strategyId} className="border-b">
                        <td className="py-2 font-medium">{profile.name}</td>
                        <td className="py-2 text-center text-green-600">{profile.sharpeRatio.toFixed(2)}</td>
                        <td className="py-2 text-center text-blue-600">{profile.sortinoRatio.toFixed(2)}</td>
                        <td className="py-2 text-center">{(profile.winRate * 100).toFixed(0)}%</td>
                        <td className="py-2 text-center text-red-600">{(profile.maxDrawdown * 100).toFixed(0)}%</td>
                        <td className="py-2 text-center text-purple-600">{profile.profitFactor.toFixed(1)}</td>
                        <td className="py-2 text-center">{(profile.expectancy * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </TabsContent>

          {/* Strategy DNA */}
          <TabsContent value="dna" className="mt-6">
            <div className="space-y-4">
              <h4 className="font-medium flex items-center">
                <Dna className="w-4 h-4 mr-2 text-purple-600" />
                Strategy DNA Analysis
              </h4>
              <div className="space-y-3">
                {strategyDNA.map((strategy) => (
                  <div key={strategy.strategyId} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className="font-medium">{strategy.name}</span>
                        <Badge variant="outline" className={getCreationMethodColor(strategy.creationMethod)}>
                          {strategy.creationMethod}
                        </Badge>
                        <Badge variant="outline" className="bg-gray-100">
                          Gen {strategy.generation}
                        </Badge>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setExpandedStrategy(
                          expandedStrategy === strategy.strategyId ? null : strategy.strategyId
                        )}
                      >
                        {expandedStrategy === strategy.strategyId ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </Button>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                      <div>
                        <span className="text-gray-600">Fitness:</span>
                        <div className="font-medium text-green-600">{strategy.fitness.toFixed(3)}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Survival:</span>
                        <div className="font-medium">{strategy.survivalTime} gens</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Offspring:</span>
                        <div className="font-medium">{strategy.offspringCount}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Parents:</span>
                        <div className="font-medium">{strategy.parentIds.length}</div>
                      </div>
                    </div>

                    {expandedStrategy === strategy.strategyId && (
                      <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                        <h5 className="font-medium mb-2">Parameters</h5>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                          {Object.entries(strategy.parameters).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-600 capitalize">{key.replace(/([A-Z])/g, ' $1')}:</span>
                              <span className="font-medium">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Intelligence Snowball */}
          <TabsContent value="intelligence" className="mt-6">
            <div className="space-y-6">
              <h4 className="font-medium flex items-center">
                <Brain className="w-4 h-4 mr-2 text-purple-600" />
                Intelligence Snowball Effect
              </h4>

              {/* Accumulated Knowledge */}
              <Card className="p-4">
                <h5 className="font-medium mb-3">Knowledge Accumulation</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-lg font-bold text-blue-600">1,247</div>
                    <div className="text-xs text-gray-600">Strategies Evolved</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-lg font-bold text-green-600">23</div>
                    <div className="text-xs text-gray-600">Market Patterns Learned</div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <div className="text-lg font-bold text-purple-600">67%</div>
                    <div className="text-xs text-gray-600">Sentiment Integration</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 rounded-lg">
                    <div className="text-lg font-bold text-orange-600">45%</div>
                    <div className="text-xs text-gray-600">News Impact Weight</div>
                  </div>
                </div>
              </Card>

              {/* Intelligence Evolution Timeline */}
              <Card className="p-4">
                <h5 className="font-medium mb-3">Intelligence Evolution Timeline</h5>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-sm">Week 1: Basic Strategy Discovery</div>
                      <div className="text-xs text-gray-600">Learned RSI and moving average patterns across 6 symbols</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-sm">Week 2: Sentiment Integration</div>
                      <div className="text-xs text-gray-600">Added news sentiment scoring, improved accuracy by 15%</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-sm">Week 3: Market Regime Adaptation</div>
                      <div className="text-xs text-gray-600">Strategies now adapt to bull/bear/high volatility conditions</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-sm">Week 4: Cross-Symbol Intelligence</div>
                      <div className="text-xs text-gray-600">Strategies share insights across SPY, QQQ, AAPL, NVDA, TSLA, BTC</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-sm">Current: Intelligence Snowball</div>
                      <div className="text-xs text-gray-600">Each evolution builds on accumulated knowledge from all previous runs</div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Future Intelligence Capabilities */}
              <Card className="p-4 border-dashed border-2 border-gray-300 bg-gray-50">
                <h5 className="font-medium mb-3 flex items-center">
                  <Zap className="w-4 h-4 mr-2 text-yellow-600" />
                  Next Intelligence Layers
                </h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="font-medium text-sm">🔮 Predictive Intelligence</div>
                    <div className="text-xs text-gray-600">Anticipate market regime changes before they happen</div>
                  </div>
                  <div className="space-y-2">
                    <div className="font-medium text-sm">🤖 Neural Strategy Evolution</div>
                    <div className="text-xs text-gray-600">Evolve neural network architectures for trading</div>
                  </div>
                  <div className="space-y-2">
                    <div className="font-medium text-sm">🌐 Multi-Asset Correlations</div>
                    <div className="text-xs text-gray-600">Learn complex relationships across all traded assets</div>
                  </div>
                  <div className="space-y-2">
                    <div className="font-medium text-sm">📈 Self-Improving Risk Models</div>
                    <div className="text-xs text-gray-600">Risk models that evolve with market conditions</div>
                  </div>
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Card>
  );
};

export default StrategyIntelligence;
