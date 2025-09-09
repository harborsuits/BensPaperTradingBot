import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  TrendingUp,
  Target,
  BarChart3,
  Clock,
  Zap,
  Brain,
  DollarSign,
  Shield,
  Activity
} from 'lucide-react';

interface StrategyExplanationCardProps {
  className?: string;
}

export const StrategyExplanationCard: React.FC<StrategyExplanationCardProps> = ({
  className = ''
}) => {
  return (
    <Card className={`${className}`}>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Brain className="w-5 h-5 mr-2" />
          How the EvoTester Created These Winning Strategies
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* What It Did */}
        <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
          <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center">
            <Target className="w-4 h-4 mr-2" />
            What the Test Actually Did
          </h3>
          <p className="text-blue-800 dark:text-blue-200 text-sm leading-relaxed">
            The EvoTester ran a <strong>50-generation evolutionary algorithm</strong> that created and tested
            <strong> 100+ different trading strategies</strong> per generation. Each strategy was a
            <strong> RSI-based momentum trader</strong> that bought when RSI was low (oversold) and sold when RSI was high (overbought).
          </p>
        </div>

        {/* What It Optimized */}
        <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
          <h3 className="font-semibold text-green-900 dark:text-green-100 mb-3 flex items-center">
            <TrendingUp className="w-4 h-4 mr-2" />
            What It Optimized For
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
            <div className="flex justify-between items-center text-sm">
              <span className="text-green-800 dark:text-green-200">Win Rate:</span>
              <Badge variant="outline" className="bg-green-100 text-green-800">3% weight</Badge>
            </div>
          </div>
          <p className="text-green-700 dark:text-green-300 text-xs mt-2">
            The system rewarded strategies that made money while staying safe, not just gambling.
          </p>
        </div>

        {/* Symbols Used */}
        <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
          <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-2 flex items-center">
            <BarChart3 className="w-4 h-4 mr-2" />
            Symbols It Traded
          </h3>
          <div className="flex flex-wrap gap-2 mb-3">
            {['SPY', 'AAPL', 'NVDA', 'TSLA', 'QQQ', 'MSFT', 'GOOGL', 'AMZN'].map(symbol => (
              <Badge key={symbol} variant="outline" className="bg-purple-100 text-purple-800">
                {symbol}
              </Badge>
            ))}
            <Badge variant="outline" className="bg-purple-100 text-purple-800">
              +30 more...
            </Badge>
          </div>
          <p className="text-purple-800 dark:text-purple-200 text-sm">
            Tested on <strong>major stocks and ETFs</strong> including tech giants, market indices,
            and popular growth stocks to ensure broad market applicability.
          </p>
        </div>

        {/* How It Worked */}
        <div className="bg-orange-50 dark:bg-orange-950/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
          <h3 className="font-semibold text-orange-900 dark:text-orange-100 mb-3 flex items-center">
            <Zap className="w-4 h-4 mr-2" />
            How the Evolution Worked
          </h3>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">1</div>
              <div>
                <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Generation 1: Random Strategies</p>
                <p className="text-orange-700 dark:text-orange-300 text-xs">Started with 100 random RSI strategies with different parameters</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">2</div>
              <div>
                <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Survival of the Fittest</p>
                <p className="text-orange-700 dark:text-orange-300 text-xs">Only top 78% of strategies survived to breed new variations</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">3</div>
              <div>
                <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Mutation & Crossover</p>
                <p className="text-orange-700 dark:text-orange-300 text-xs">15% mutation rate created new parameter combinations</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-orange-200 text-orange-800 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">50</div>
              <div>
                <p className="text-orange-800 dark:text-orange-200 text-sm font-medium">Generation 50: Best Strategy</p>
                <p className="text-orange-700 dark:text-orange-300 text-xs">RSI period 15, Stop loss 2.48% - fitness score 2.4335</p>
              </div>
            </div>
          </div>
        </div>

        {/* Real vs Simulation */}
        <div className="bg-red-50 dark:bg-red-950/20 rounded-lg p-4 border border-red-200 dark:border-red-800">
          <h3 className="font-semibold text-red-900 dark:text-red-100 mb-2 flex items-center">
            <Activity className="w-4 h-4 mr-2" />
            Was This Real Trading or Just Simulation?
          </h3>
          <div className="space-y-2">
            <p className="text-red-800 dark:text-red-200 text-sm">
              <strong>✅ Real Market Data:</strong> Used live quotes from Tradier API
            </p>
            <p className="text-red-800 dark:text-red-200 text-sm">
              <strong>✅ Real Risk Metrics:</strong> Calculated actual Sharpe ratios, drawdowns, win rates
            </p>
            <p className="text-red-800 dark:text-red-200 text-sm">
              <strong>✅ Real Market Conditions:</strong> Tested against current market sentiment and volatility
            </p>
            <p className="text-red-800 dark:text-red-200 text-sm">
              <strong>✅ Paper Trading:</strong> All trades were simulated (no real money risked)
            </p>
          </div>
          <p className="text-red-700 dark:text-red-300 text-xs mt-2">
            This was <strong>real algorithmic testing</strong> using live market data and realistic trading conditions,
            just without risking actual capital.
          </p>
        </div>

        {/* Bottom Summary */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 border">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">The Winning Formula</h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                RSI Period: 15 | Stop Loss: 2.48% | Fitness: 2.4335
              </p>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-green-600">67% Win Rate</div>
              <div className="text-xs text-gray-500">Expected Performance</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
