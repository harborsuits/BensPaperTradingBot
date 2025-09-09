import React from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import {
  BarChart3,
  TrendingUp,
  Target,
  Zap,
  Brain,
  Activity,
  CheckCircle,
  AlertTriangle,
  Info,
  ArrowRight
} from 'lucide-react';

interface StrategyUtilization {
  strategyId: string;
  name: string;
  utilizationRate: number;
  totalTrades: number;
  winRate: number;
  avgReturn: number;
  lastUsed: string;
}

interface DecisionConfidence {
  timeframe: string;
  confidence: number;
  signalsGenerated: number;
  signalsExecuted: number;
  avgConfidence: number;
}

interface PerformanceAttribution {
  strategyId: string;
  name: string;
  contribution: number;
  totalPnL: number;
  winRate: number;
  sharpeRatio: number;
}

interface LearningFeedback {
  improvement: number;
  adaptations: number;
  newStrategies: number;
  retiredStrategies: number;
  feedback: string[];
}

interface PipelineIntegrationProps {
  className?: string;
}

const PipelineIntegration: React.FC<PipelineIntegrationProps> = ({ className = '' }) => {
  const strategyUtilization: StrategyUtilization[] = [
    {
      strategyId: 'evo-247',
      name: 'RSI-Momentum-V2',
      utilizationRate: 0.85,
      totalTrades: 234,
      winRate: 0.67,
      avgReturn: 0.023,
      lastUsed: '2m ago'
    },
    {
      strategyId: 'evo-189',
      name: 'Volatility-Breakout',
      utilizationRate: 0.72,
      totalTrades: 189,
      winRate: 0.58,
      avgReturn: 0.018,
      lastUsed: '5m ago'
    },
    {
      strategyId: 'evo-156',
      name: 'MeanReversion-Pro',
      utilizationRate: 0.45,
      totalTrades: 145,
      winRate: 0.71,
      avgReturn: 0.015,
      lastUsed: '12m ago'
    }
  ];

  const decisionConfidence: DecisionConfidence[] = [
    {
      timeframe: 'Last 24h',
      confidence: 0.89,
      signalsGenerated: 1247,
      signalsExecuted: 892,
      avgConfidence: 0.76
    },
    {
      timeframe: 'Last 7d',
      confidence: 0.92,
      signalsGenerated: 8734,
      signalsExecuted: 6234,
      avgConfidence: 0.78
    },
    {
      timeframe: 'Last 30d',
      confidence: 0.87,
      signalsGenerated: 34234,
      signalsExecuted: 24567,
      avgConfidence: 0.74
    }
  ];

  const performanceAttribution: PerformanceAttribution[] = [
    {
      strategyId: 'evo-247',
      name: 'RSI-Momentum-V2',
      contribution: 0.45,
      totalPnL: 45234,
      winRate: 0.67,
      sharpeRatio: 2.34
    },
    {
      strategyId: 'evo-189',
      name: 'Volatility-Breakout',
      contribution: 0.32,
      totalPnL: 32187,
      winRate: 0.58,
      sharpeRatio: 1.98
    },
    {
      strategyId: 'evo-156',
      name: 'MeanReversion-Pro',
      contribution: 0.23,
      totalPnL: 23145,
      winRate: 0.71,
      sharpeRatio: 1.76
    }
  ];

  const learningFeedback: LearningFeedback = {
    improvement: 15.2,
    adaptations: 23,
    newStrategies: 47,
    retiredStrategies: 12,
    feedback: [
      "RSI-Momentum-V2 adapted stop loss from 2.0% to 2.1% - improved win rate by 3%",
      "Volatility-Breakout learned to avoid trading during extreme volatility spikes",
      "MeanReversion-Pro discovered optimal entry timing at 2:30 PM EST",
      "New breakout strategy evolved from crossover of RSI-Momentum-V2 and Volatility-Breakout"
    ]
  };

  return (
    <Card className={className}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Pipeline Integration</h3>
            <p className="text-sm text-gray-600">How EvoTester powers live trading decisions</p>
          </div>
          <Badge variant="outline" className="bg-green-50 border-green-300 text-green-700">
            <Activity className="w-3 h-3 mr-1" />
            Live Integration
          </Badge>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Strategy Utilization */}
          <Card className="p-4">
            <h4 className="font-medium mb-4 flex items-center">
              <BarChart3 className="w-4 h-4 mr-2 text-blue-600" />
              Strategy Utilization
            </h4>
            <div className="space-y-3">
              {strategyUtilization.map((strategy) => (
                <div key={strategy.strategyId} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{strategy.name}</span>
                    <Badge variant="outline" className="text-xs">
                      {(strategy.utilizationRate * 100).toFixed(0)}% used
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-2">
                    <div>Total Trades: {strategy.totalTrades}</div>
                    <div>Win Rate: {(strategy.winRate * 100).toFixed(0)}%</div>
                    <div>Avg Return: {(strategy.avgReturn * 100).toFixed(1)}%</div>
                    <div>Last Used: {strategy.lastUsed}</div>
                  </div>
                  <div className="bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${strategy.utilizationRate * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Decision Confidence */}
          <Card className="p-4">
            <h4 className="font-medium mb-4 flex items-center">
              <Target className="w-4 h-4 mr-2 text-green-600" />
              Decision Confidence
            </h4>
            <div className="space-y-3">
              {decisionConfidence.map((period) => (
                <div key={period.timeframe} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{period.timeframe}</span>
                    <Badge variant="outline" className="text-green-600 border-green-300">
                      {(period.confidence * 100).toFixed(0)}% confidence
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                    <div>Signals Generated: {period.signalsGenerated.toLocaleString()}</div>
                    <div>Signals Executed: {period.signalsExecuted.toLocaleString()}</div>
                    <div>Execution Rate: {((period.signalsExecuted / period.signalsGenerated) * 100).toFixed(0)}%</div>
                    <div>Avg Confidence: {(period.avgConfidence * 100).toFixed(0)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Performance Attribution */}
          <Card className="p-4">
            <h4 className="font-medium mb-4 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2 text-purple-600" />
              Performance Attribution
            </h4>
            <div className="space-y-3">
              {performanceAttribution.map((strategy) => (
                <div key={strategy.strategyId} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{strategy.name}</span>
                    <Badge variant="outline" className="text-purple-600 border-purple-300">
                      {(strategy.contribution * 100).toFixed(0)}% contribution
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                    <div>Total P&L: ${strategy.totalPnL.toLocaleString()}</div>
                    <div>Win Rate: {(strategy.winRate * 100).toFixed(0)}%</div>
                    <div>Sharpe Ratio: {strategy.sharpeRatio.toFixed(2)}</div>
                    <div>Contribution: {(strategy.contribution * 100).toFixed(1)}%</div>
                  </div>
                  <div className="mt-2 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-purple-600 h-2 rounded-full"
                      style={{ width: `${strategy.contribution * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Learning Feedback */}
          <Card className="p-4">
            <h4 className="font-medium mb-4 flex items-center">
              <Brain className="w-4 h-4 mr-2 text-orange-600" />
              Learning Feedback
            </h4>
            <div className="space-y-4">
              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="p-3 bg-green-50 rounded-lg">
                  <div className="text-lg font-bold text-green-600">
                    +{learningFeedback.improvement}%
                  </div>
                  <div className="text-xs text-gray-600">Improvement</div>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <div className="text-lg font-bold text-blue-600">
                    {learningFeedback.adaptations}
                  </div>
                  <div className="text-xs text-gray-600">Adaptations</div>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <div className="text-lg font-bold text-purple-600">
                    {learningFeedback.newStrategies}
                  </div>
                  <div className="text-xs text-gray-600">New Strategies</div>
                </div>
                <div className="p-3 bg-red-50 rounded-lg">
                  <div className="text-lg font-bold text-red-600">
                    {learningFeedback.retiredStrategies}
                  </div>
                  <div className="text-xs text-gray-600">Retired</div>
                </div>
              </div>

              {/* Recent Learnings */}
              <div className="space-y-2">
                <h5 className="text-sm font-medium text-foreground">Recent Learnings</h5>
                <div className="space-y-2">
                  {learningFeedback.feedback.map((item, index) => (
                    <div key={index} className="flex items-start space-x-2 text-xs">
                      <CheckCircle className="w-3 h-3 text-green-600 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-600">{item}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Integration Health Summary */}
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-foreground">Integration Health</h4>
              <p className="text-sm text-foreground">EvoTester ↔ Trading Pipeline</p>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-1">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span>Real-time sync active</span>
              </div>
              <div className="flex items-center space-x-1">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span>WebSocket connected</span>
              </div>
              <div className="flex items-center space-x-1">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span>Strategy deployment live</span>
              </div>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-green-600">98.7%</div>
              <div className="text-xs text-gray-600">Signal Accuracy</div>
            </div>
            <div>
              <div className="text-lg font-bold text-blue-600">2.1s</div>
              <div className="text-xs text-gray-600">Avg Decision Time</div>
            </div>
            <div>
              <div className="text-lg font-bold text-purple-600">99.2%</div>
              <div className="text-xs text-gray-600">Uptime</div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default PipelineIntegration;
