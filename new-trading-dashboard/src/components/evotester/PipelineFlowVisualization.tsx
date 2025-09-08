import React from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { CheckCircle, Clock, AlertCircle, ArrowRight, Brain, Database, Target, Shield, MapPin, Settings, TrendingUp } from 'lucide-react';

interface PipelineStage {
  name: string;
  icon: React.ComponentType<any>;
  status: 'active' | 'waiting' | 'completed';
  evoContribution: {
    strategiesUsed: number;
    fitnessThreshold: number;
    lastUpdate: string;
  };
  metrics: {
    throughput: number;
    successRate: number;
    avgLatency: number;
  };
}

const PipelineFlowVisualization: React.FC = () => {
  const pipelineStages: PipelineStage[] = [
    {
      name: 'INGEST',
      icon: Database,
      status: 'completed',
      evoContribution: {
        strategiesUsed: 0,
        fitnessThreshold: 0,
        lastUpdate: '2s ago'
      },
      metrics: {
        throughput: 1250,
        successRate: 99.2,
        avgLatency: 3.2
      }
    },
    {
      name: 'CONTEXT',
      icon: Brain,
      status: 'active',
      evoContribution: {
        strategiesUsed: 45,
        fitnessThreshold: 1.8,
        lastUpdate: '1s ago'
      },
      metrics: {
        throughput: 890,
        successRate: 95.1,
        avgLatency: 12.5
      }
    },
    {
      name: 'CANDIDATES',
      icon: Target,
      status: 'active',
      evoContribution: {
        strategiesUsed: 12,
        fitnessThreshold: 2.1,
        lastUpdate: '3s ago'
      },
      metrics: {
        throughput: 234,
        successRate: 87.3,
        avgLatency: 45.8
      }
    },
    {
      name: 'GATES',
      icon: Shield,
      status: 'waiting',
      evoContribution: {
        strategiesUsed: 8,
        fitnessThreshold: 2.3,
        lastUpdate: '5s ago'
      },
      metrics: {
        throughput: 89,
        successRate: 92.4,
        avgLatency: 23.1
      }
    },
    {
      name: 'PLAN',
      icon: Settings,
      status: 'waiting',
      evoContribution: {
        strategiesUsed: 3,
        fitnessThreshold: 2.4,
        lastUpdate: '12s ago'
      },
      metrics: {
        throughput: 34,
        successRate: 94.7,
        avgLatency: 67.3
      }
    },
    {
      name: 'ROUTE',
      icon: MapPin,
      status: 'waiting',
      evoContribution: {
        strategiesUsed: 1,
        fitnessThreshold: 2.5,
        lastUpdate: '18s ago'
      },
      metrics: {
        throughput: 12,
        successRate: 96.2,
        avgLatency: 89.4
      }
    },
    {
      name: 'MANAGE',
      icon: TrendingUp,
      status: 'waiting',
      evoContribution: {
        strategiesUsed: 1,
        fitnessThreshold: 2.5,
        lastUpdate: '25s ago'
      },
      metrics: {
        throughput: 8,
        successRate: 98.1,
        avgLatency: 45.6
      }
    },
    {
      name: 'LEARN',
      icon: Brain,
      status: 'waiting',
      evoContribution: {
        strategiesUsed: 0,
        fitnessThreshold: 0,
        lastUpdate: '1m ago'
      },
      metrics: {
        throughput: 3,
        successRate: 100,
        avgLatency: 120.0
      }
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'waiting':
        return <Clock className="w-4 h-4 text-gray-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 border-green-300 text-green-800';
      case 'completed':
        return 'bg-blue-100 border-blue-300 text-blue-800';
      case 'waiting':
        return 'bg-gray-100 border-gray-300 text-gray-600';
      default:
        return 'bg-gray-100 border-gray-300 text-gray-600';
    }
  };

  return (
    <Card>
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Pipeline Flow</h3>
          <Badge variant="outline" className="bg-purple-50 border-purple-300 text-purple-700">
            <Brain className="w-3 h-3 mr-1" />
            EvoTester Integrated
          </Badge>
        </div>

        <div className="space-y-3">
          {pipelineStages.map((stage, index) => {
            const IconComponent = stage.icon;
            return (
              <div key={stage.name} className="relative">
                {/* Connection line */}
                {index < pipelineStages.length - 1 && (
                  <div className="absolute left-6 top-12 w-0.5 h-8 bg-gray-200 z-0" />
                )}

                <div className={`flex items-start space-x-4 p-4 rounded-lg border ${getStatusColor(stage.status)}`}>
                  {/* Status indicator */}
                  <div className="flex-shrink-0 mt-1">
                    {getStatusIcon(stage.status)}
                  </div>

                  {/* Stage icon */}
                  <div className="flex-shrink-0">
                    <IconComponent className="w-8 h-8 text-gray-600" />
                  </div>

                  {/* Stage info */}
                  <div className="flex-grow">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">{stage.name}</h4>
                      <span className="text-xs text-gray-500">{stage.evoContribution.lastUpdate}</span>
                    </div>

                    {/* EvoTester contribution */}
                    {stage.evoContribution.strategiesUsed > 0 && (
                      <div className="mb-2 p-2 bg-purple-50 rounded border border-purple-200">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-purple-700 font-medium">
                            🧬 {stage.evoContribution.strategiesUsed} strategies used
                          </span>
                          <span className="text-purple-600">
                            Fitness ≥ {stage.evoContribution.fitnessThreshold}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-4 text-xs">
                      <div>
                        <span className="text-gray-500">Throughput:</span>
                        <div className="font-medium">{stage.metrics.throughput}/min</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Success:</span>
                        <div className="font-medium">{stage.metrics.successRate}%</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Latency:</span>
                        <div className="font-medium">{stage.metrics.avgLatency}s</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Summary stats */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium mb-2">Evolution Impact Summary</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Strategies Deployed:</span>
              <div className="font-medium text-green-600">247 active</div>
            </div>
            <div>
              <span className="text-gray-600">Avg Fitness:</span>
              <div className="font-medium text-blue-600">2.34</div>
            </div>
            <div>
              <span className="text-gray-600">Success Rate:</span>
              <div className="font-medium text-purple-600">94.7%</div>
            </div>
            <div>
              <span className="text-gray-600">Improvement:</span>
              <div className="font-medium text-orange-600">+15.2% this week</div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default PipelineFlowVisualization;
