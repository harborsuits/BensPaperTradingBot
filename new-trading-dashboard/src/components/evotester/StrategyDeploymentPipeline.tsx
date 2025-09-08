import React from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Play, Pause, Square, TrendingUp, AlertTriangle, CheckCircle, Clock, ArrowRight, BarChart3 } from 'lucide-react';

interface DeploymentStage {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'failed' | 'paused';
  strategies: number;
  successRate: number;
  avgDuration: string;
  lastActivity: string;
}

interface StrategyDeploymentPipelineProps {
  className?: string;
}

const StrategyDeploymentPipeline: React.FC<StrategyDeploymentPipelineProps> = ({ className = '' }) => {
  const deploymentStages: DeploymentStage[] = [
    {
      id: 'evolution',
      name: 'Evolution',
      description: 'Strategies evolving through genetic algorithms',
      status: 'active',
      strategies: 1247,
      successRate: 94.2,
      avgDuration: '45 min',
      lastActivity: '2s ago'
    },
    {
      id: 'validation',
      name: 'Backtest Validation',
      description: 'Testing strategies on historical data',
      status: 'active',
      strategies: 89,
      successRate: 87.3,
      avgDuration: '12 min',
      lastActivity: '1m ago'
    },
    {
      id: 'paper_trading',
      name: 'Paper Trading',
      description: 'Risk-free live simulation',
      status: 'active',
      strategies: 23,
      successRate: 91.7,
      avgDuration: '2 weeks',
      lastActivity: '5m ago'
    },
    {
      id: 'live_trading',
      name: 'Live Trading',
      description: 'Real money deployment',
      status: 'completed',
      strategies: 8,
      successRate: 95.4,
      avgDuration: '6 weeks',
      lastActivity: '2h ago'
    },
    {
      id: 'monitoring',
      name: 'Performance Monitoring',
      description: 'Continuous performance tracking',
      status: 'active',
      strategies: 8,
      successRate: 92.1,
      avgDuration: 'Ongoing',
      lastActivity: '30s ago'
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-600" />;
      case 'failed':
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'border-green-300 bg-green-50';
      case 'completed':
        return 'border-blue-300 bg-blue-50';
      case 'paused':
        return 'border-yellow-300 bg-yellow-50';
      case 'failed':
        return 'border-red-300 bg-red-50';
      default:
        return 'border-gray-300 bg-gray-50';
    }
  };

  const getActionButton = (stage: DeploymentStage) => {
    switch (stage.status) {
      case 'active':
        return (
          <Button size="sm" variant="outline" className="text-yellow-600 border-yellow-300 hover:bg-yellow-50">
            <Pause className="w-3 h-3 mr-1" />
            Pause
          </Button>
        );
      case 'paused':
        return (
          <Button size="sm" variant="outline" className="text-green-600 border-green-300 hover:bg-green-50">
            <Play className="w-3 h-3 mr-1" />
            Resume
          </Button>
        );
      case 'failed':
        return (
          <Button size="sm" variant="outline" className="text-red-600 border-red-300 hover:bg-red-50">
            <Square className="w-3 h-3 mr-1" />
            Retry
          </Button>
        );
      default:
        return null;
    }
  };

  return (
    <Card className={className}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Strategy Deployment Pipeline</h3>
            <p className="text-sm text-gray-600">From evolution to live trading</p>
          </div>
          <Badge variant="outline" className="bg-green-50 border-green-300 text-green-700">
            <TrendingUp className="w-3 h-3 mr-1" />
            Auto-deployment enabled
          </Badge>
        </div>

        <div className="space-y-4">
          {deploymentStages.map((stage, index) => (
            <div key={stage.id} className="relative">
              {/* Connection arrow */}
              {index < deploymentStages.length - 1 && (
                <div className="absolute left-6 top-16 flex items-center justify-center w-4 h-4 z-10">
                  <ArrowRight className="w-4 h-4 text-gray-400" />
                </div>
              )}

              <div className={`border-2 rounded-lg p-4 transition-all duration-200 ${getStatusColor(stage.status)}`}>
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    {/* Status indicator */}
                    <div className="flex-shrink-0 mt-1">
                      {getStatusIcon(stage.status)}
                    </div>

                    {/* Stage info */}
                    <div className="flex-grow">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="font-medium text-gray-900">{stage.name}</h4>
                        <Badge
                          variant={stage.status === 'active' ? 'default' : 'outline'}
                          className={stage.status === 'active' ? 'bg-green-100 text-green-800' : ''}
                        >
                          {stage.status}
                        </Badge>
                      </div>

                      <p className="text-sm text-gray-600 mb-3">{stage.description}</p>

                      {/* Metrics */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Strategies:</span>
                          <div className="font-medium">{stage.strategies}</div>
                        </div>
                        <div>
                          <span className="text-gray-500">Success Rate:</span>
                          <div className="font-medium text-green-600">{stage.successRate}%</div>
                        </div>
                        <div>
                          <span className="text-gray-500">Avg Duration:</span>
                          <div className="font-medium">{stage.avgDuration}</div>
                        </div>
                        <div>
                          <span className="text-gray-500">Last Activity:</span>
                          <div className="font-medium">{stage.lastActivity}</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Action button */}
                  <div className="flex-shrink-0">
                    {getActionButton(stage)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Deployment summary */}
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-gray-900">Pipeline Health</h4>
              <p className="text-sm text-gray-600">Overall system performance</p>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div className="text-center">
                <div className="text-lg font-bold text-green-600">94.2%</div>
                <div className="text-gray-500">Success Rate</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">2.34</div>
                <div className="text-gray-500">Avg Fitness</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600">8</div>
                <div className="text-gray-500">Live Strategies</div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick actions */}
        <div className="mt-4 flex justify-between items-center">
          <div className="text-sm text-gray-600">
            Last successful deployment: Strategy #247 → Live Trading (2 hours ago)
          </div>
          <div className="flex space-x-2">
            <Button size="sm" variant="outline">
              <BarChart3 className="w-3 h-3 mr-1" />
              View Analytics
            </Button>
            <Button size="sm">
              Configure Pipeline
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default StrategyDeploymentPipeline;
