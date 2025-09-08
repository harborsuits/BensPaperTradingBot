import React from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Activity, TrendingUp, Brain, Zap } from 'lucide-react';

interface EvolutionStatusBarProps {
  activeSessions: number;
  totalStrategies: number;
  bestFitness: number;
  marketRegime: string;
  lastDeployment: Date;
  activeSymbols?: string[];
  sentimentScore?: number;
  newsImpactScore?: number;
}

const EvolutionStatusBar: React.FC<EvolutionStatusBarProps> = ({
  activeSessions,
  totalStrategies,
  bestFitness,
  marketRegime,
  lastDeployment,
  activeSymbols = ['SPY', 'QQQ', 'AAPL'],
  sentimentScore = 0.67,
  newsImpactScore = 0.45
}) => {
  const improvementRate = 5.2; // This would come from real data
  const timeSinceDeployment = Math.floor((Date.now() - lastDeployment.getTime()) / (1000 * 60 * 60));

  return (
    <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-blue-600" />
            <span className="font-semibold text-gray-800">Evolution Status</span>
          </div>

          <div className="flex items-center space-x-4 text-sm">
            {/* Active Sessions */}
            <div className="flex items-center space-x-1">
              <Activity className="h-4 w-4 text-green-600" />
              <span className="text-gray-600">{activeSessions} sessions</span>
            </div>

            {/* Total Strategies */}
            <div className="flex items-center space-x-1">
              <span className="text-gray-600">{totalStrategies.toLocaleString()} strategies</span>
            </div>

            {/* Best Fitness */}
            <div className="flex items-center space-x-1">
              <TrendingUp className="h-4 w-4 text-blue-600" />
              <span className="font-medium text-blue-600">Best: {bestFitness.toFixed(3)}</span>
              <Badge variant="outline" className="text-green-600 border-green-300 bg-green-50">
                +{improvementRate}%
              </Badge>
            </div>

            {/* Market Regime */}
            <div className="flex items-center space-x-1">
              <Zap className="h-4 w-4 text-purple-600" />
              <span className="text-gray-600">{marketRegime}</span>
            </div>

            {/* Active Symbols */}
            <div className="flex items-center space-x-1">
              <span className="text-gray-600">
                Symbols: {activeSymbols.slice(0, 3).join(', ')}{activeSymbols.length > 3 ? '...' : ''}
              </span>
            </div>

            {/* Sentiment & News */}
            <div className="flex items-center space-x-1">
              <Brain className="w-3 h-3 text-purple-600" />
              <span className="text-gray-600">
                Sentiment: {(sentimentScore * 100).toFixed(0)}%
              </span>
              <Badge variant="outline" className="text-blue-600 border-blue-300">
                News: {(newsImpactScore * 100).toFixed(0)}%
              </Badge>
            </div>

            {/* Last Deployment */}
            <div className="flex items-center space-x-1">
              <span className="text-gray-600">
                Deployed: {timeSinceDeployment}h ago
              </span>
              <Badge variant="outline" className="text-green-600 border-green-300">
                Intelligence: ON
              </Badge>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default EvolutionStatusBar;
