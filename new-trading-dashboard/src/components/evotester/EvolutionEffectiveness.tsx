import React from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TrendingUp, TrendingDown, Target, BarChart3, Zap, Award, AlertTriangle, CheckCircle } from 'lucide-react';

interface EvolutionMetrics {
  fitnessImprovement: {
    rate: number;
    trend: 'up' | 'down' | 'stable';
    period: string;
  };
  strategySurvival: {
    rate: number;
    totalStrategies: number;
    survivingStrategies: number;
  };
  marketAdaptation: {
    score: number;
    regime: string;
    adaptationSpeed: string;
  };
  backtestVsLive: {
    backtestPerformance: number;
    livePerformance: number;
    degradation: number;
    confidence: number;
  };
}

interface EvolutionEffectivenessProps {
  className?: string;
}

const EvolutionEffectiveness: React.FC<EvolutionEffectivenessProps> = ({ className = '' }) => {
  const metrics: EvolutionMetrics = {
    fitnessImprovement: {
      rate: 5.2,
      trend: 'up',
      period: '24h'
    },
    strategySurvival: {
      rate: 12.3,
      totalStrategies: 1247,
      survivingStrategies: 153
    },
    marketAdaptation: {
      score: 87.3,
      regime: 'Bull Market',
      adaptationSpeed: '2.3 generations'
    },
    backtestVsLive: {
      backtestPerformance: 2.34,
      livePerformance: 2.18,
      degradation: 7.1,
      confidence: 89.2
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      default:
        return <BarChart3 className="w-4 h-4 text-gray-600" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'down':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const MetricCard: React.FC<{
    title: string;
    value: string | number;
    subtitle: string;
    trend?: string;
    icon: React.ComponentType<any>;
    status?: 'good' | 'warning' | 'danger';
  }> = ({ title, value, subtitle, trend, icon: IconComponent, status = 'good' }) => {
    const getStatusColor = () => {
      switch (status) {
        case 'good':
          return 'border-green-200 bg-green-50';
        case 'warning':
          return 'border-yellow-200 bg-yellow-50';
        case 'danger':
          return 'border-red-200 bg-red-50';
        default:
          return 'border-gray-200 bg-gray-50';
      }
    };

    return (
      <Card className={`p-4 ${getStatusColor()}`}>
        <div className="flex items-center justify-between mb-2">
          <IconComponent className="w-5 h-5 text-gray-600" />
          {trend && (
            <Badge variant="outline" className={getTrendColor(trend)}>
              {getTrendIcon(trend)}
              <span className="ml-1">{trend.toUpperCase()}</span>
            </Badge>
          )}
        </div>
        <div className="space-y-1">
          <h3 className="font-medium text-foreground">{title}</h3>
          <div className="text-2xl font-bold text-foreground">{value}</div>
          <p className="text-sm text-foreground">{subtitle}</p>
        </div>
      </Card>
    );
  };

  return (
    <Card className={className}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Evolution Effectiveness</h3>
            <p className="text-sm text-gray-600">How well the evolutionary process is working</p>
          </div>
          <Badge variant="outline" className="bg-blue-50 border-blue-300 text-blue-700">
            <Zap className="w-3 h-3 mr-1" />
            Real-time monitoring
          </Badge>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Fitness Improvement"
            value={`${metrics.fitnessImprovement.rate}%`}
            subtitle={`Over last ${metrics.fitnessImprovement.period}`}
            trend={metrics.fitnessImprovement.trend}
            icon={TrendingUp}
            status="good"
          />

          <MetricCard
            title="Strategy Survival"
            value={`${metrics.strategySurvival.rate}%`}
            subtitle={`${metrics.strategySurvival.survivingStrategies}/${metrics.strategySurvival.totalStrategies} survive`}
            icon={Target}
            status="warning"
          />

          <MetricCard
            title="Market Adaptation"
            value={`${metrics.marketAdaptation.score}%`}
            subtitle={`${metrics.marketAdaptation.regime} • ${metrics.marketAdaptation.adaptationSpeed}`}
            icon={Zap}
            status="good"
          />

          <MetricCard
            title="Backtest vs Live"
            value={`${metrics.backtestVsLive.degradation}% degradation`}
            subtitle={`Confidence: ${metrics.backtestVsLive.confidence}%`}
            icon={BarChart3}
            status={metrics.backtestVsLive.degradation > 10 ? 'warning' : 'good'}
          />
        </div>

        {/* Detailed Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Fitness Improvement Details */}
          <Card className="p-4">
            <h4 className="font-medium mb-3 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2 text-green-600" />
              Fitness Improvement Analysis
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Current Generation:</span>
                <span className="font-medium">47</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Best Fitness:</span>
                <span className="font-medium text-green-600">2.34</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Improvement Rate:</span>
                <span className="font-medium text-green-600">+5.2%/day</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Generations to Convergence:</span>
                <span className="font-medium">~23 remaining</span>
              </div>
            </div>
          </Card>

          {/* Strategy Survival Details */}
          <Card className="p-4">
            <h4 className="font-medium mb-3 flex items-center">
              <Target className="w-4 h-4 mr-2 text-blue-600" />
              Strategy Survival Analysis
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Population Size:</span>
                <span className="font-medium">1,247 strategies</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Survival Rate:</span>
                <span className="font-medium text-blue-600">12.3%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Elite Strategies:</span>
                <span className="font-medium text-green-600">153 surviving</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Rejection Rate:</span>
                <span className="font-medium text-red-600">87.7%</span>
              </div>
            </div>
          </Card>

          {/* Market Adaptation Details */}
          <Card className="p-4">
            <h4 className="font-medium mb-3 flex items-center">
              <Zap className="w-4 h-4 mr-2 text-purple-600" />
              Market Adaptation Analysis
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Current Regime:</span>
                <span className="font-medium">Bull Market</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Adaptation Score:</span>
                <span className="font-medium text-purple-600">87.3%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Response Time:</span>
                <span className="font-medium">2.3 generations</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Regime Switches:</span>
                <span className="font-medium text-blue-600">3 this week</span>
              </div>
            </div>
          </Card>

          {/* Backtest vs Live Details */}
          <Card className="p-4">
            <h4 className="font-medium mb-3 flex items-center">
              <BarChart3 className="w-4 h-4 mr-2 text-orange-600" />
              Backtest vs Live Analysis
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Backtest Performance:</span>
                <span className="font-medium">2.34 Sharpe</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Live Performance:</span>
                <span className="font-medium">2.18 Sharpe</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Degradation:</span>
                <span className="font-medium text-orange-600">7.1%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Confidence Level:</span>
                <span className="font-medium text-green-600">89.2%</span>
              </div>
            </div>
          </Card>
        </div>

        {/* Health Indicators */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium mb-3">System Health Indicators</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span className="text-sm">Evolution Process: Healthy</span>
            </div>
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
              <span className="text-sm">Strategy Survival: Monitor</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span className="text-sm">Market Adaptation: Good</span>
            </div>
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
              <span className="text-sm">Live Degradation: Watch</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default EvolutionEffectiveness;
