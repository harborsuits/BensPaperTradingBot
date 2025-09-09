/**
 * ============================================
 * [CARD: EVOLUTION LIFECYCLE VIEW]
 * Timeline analysis, champion lineage tracking, population dynamics
 * ============================================
 */

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import {
  TrendingUp,
  TrendingDown,
  Award,
  Target,
  Clock,
  Zap,
  Users,
  BarChart3,
  Activity,
  GitBranch,
  Crown,
  Flame,
  Star,
  RefreshCw,
  Loader2
} from 'lucide-react';
import TimeSeriesChart from '@/components/ui/TimeSeriesChart';
import { evoTesterApi } from '@/services/api';

interface GenerationData {
  generation: number;
  populationSize: number;
  bestFitness: number;
  avgFitness: number;
  diversityScore: number;
  championStrategy: string;
  survivalRate: number;
  mutationRate: number;
  timestamp: string;
  promotedStrategies: number;
  extinctStrategies: number;
}

interface ChampionLineage {
  strategyId: string;
  name: string;
  fitness: number;
  generationsSurvived: number;
  firstAppearance: number;
  lastAppearance: number;
  peakFitness: number;
  traits: string[];
  evolutionPath: number[];
}

interface EvoLifecycleViewProps {
  sessionId?: string;
  className?: string;
}

const EvoLifecycleView: React.FC<EvoLifecycleViewProps> = ({
  sessionId,
  className = ''
}) => {
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'timeline' | 'lineage' | 'diversity'>('timeline');

  // Real-time data connections
  const { data: sessionStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['evoTester', 'status', sessionId],
    queryFn: () => sessionId ? evoTesterApi.getEvoStatus(sessionId) : null,
    enabled: !!sessionId,
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000,
  });

  const { data: generationsData, isLoading: generationLoading } = useQuery({
    queryKey: ['evoTester', 'generations', sessionId],
    queryFn: () => sessionId ? evoTesterApi.getGenerations(sessionId) : null,
    enabled: !!sessionId,
    refetchInterval: 15000, // Refresh every 15 seconds
    staleTime: 7500,
  });

  // Use live generation data with fallback to empty array
  const liveGenerationsData = generationsData?.generations || [];

  // Generate champion lineage from real generation data
  const championLineage: ChampionLineage[] = React.useMemo(() => {
    if (liveGenerationsData.length === 0) return [];

    // Find the best performing strategies across generations
    const strategyMap = new Map<string, ChampionLineage>();

    liveGenerationsData.forEach((gen, index) => {
      const strategyId = gen.championStrategy;
      const fitness = gen.bestFitness;

      if (!strategyMap.has(strategyId)) {
        strategyMap.set(strategyId, {
          strategyId,
          name: strategyId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          fitness,
          generationsSurvived: 1,
          firstAppearance: gen.generation,
          lastAppearance: gen.generation,
          peakFitness: fitness,
          traits: ['Technical', 'Adaptive'], // Default traits
          evolutionPath: [gen.generation]
        });
      } else {
        const existing = strategyMap.get(strategyId)!;
        existing.generationsSurvived++;
        existing.lastAppearance = gen.generation;
        existing.peakFitness = Math.max(existing.peakFitness, fitness);
        existing.evolutionPath.push(gen.generation);

        // Update fitness to latest
        existing.fitness = fitness;
      }
    });

    // Convert to array and sort by peak fitness
    return Array.from(strategyMap.values())
      .sort((a, b) => b.peakFitness - a.peakFitness)
      .slice(0, 5); // Top 5 champions
  }, [liveGenerationsData]);

  const formatChartData = () => {
    return liveGenerationsData.map(gen => ({
      timestamp: gen.timestamp,
      generation: gen.generation,
      bestFitness: gen.bestFitness,
      avgFitness: gen.avgFitness,
      diversityScore: gen.diversityScore,
      populationSize: gen.populationSize,
      survivalRate: gen.survivalRate * 100
    }));
  };

  const getFitnessTrend = () => {
    if (liveGenerationsData.length < 2) return 'stable';
    const latest = liveGenerationsData[liveGenerationsData.length - 1].bestFitness;
    const previous = liveGenerationsData[liveGenerationsData.length - 2].bestFitness;
    const change = ((latest - previous) / previous) * 100;

    if (change > 5) return 'improving';
    if (change < -5) return 'declining';
    return 'stable';
  };

  const getFitnessTrendIcon = () => {
    const trend = getFitnessTrend();
    switch (trend) {
      case 'improving':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'declining':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const renderTimelineView = () => (
    <div className="space-y-4">
      {/* Generation Timeline */}
      <div className="space-y-3">
        {liveGenerationsData.map((gen, index) => (
          <div
            key={gen.generation}
            className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
              selectedGeneration === gen.generation
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}
            onClick={() => setSelectedGeneration(
              selectedGeneration === gen.generation ? null : gen.generation
            )}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <Badge variant="outline" className="bg-purple-50 border-purple-300 text-purple-700">
                    <GitBranch className="w-3 h-3 mr-1" />
                    Gen {gen.generation}
                  </Badge>
                  <span className="text-sm text-gray-500">
                    {new Date(gen.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                {gen.promotedStrategies > 0 && (
                  <Badge className="bg-green-100 text-green-800">
                    <Crown className="w-3 h-3 mr-1" />
                    {gen.promotedStrategies} Promoted
                  </Badge>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-blue-600">
                  Fitness: {gen.bestFitness.toFixed(3)}
                </span>
                <span className="text-sm text-gray-600">
                  Pop: {gen.populationSize}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-green-500" />
                <span>Champion: {gen.championStrategy}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-blue-500" />
                <span>Survival: {(gen.survivalRate * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4 text-purple-500" />
                <span>Diversity: {(gen.diversityScore * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-orange-500" />
                <span>Mutation: {(gen.mutationRate * 100).toFixed(1)}%</span>
              </div>
            </div>

            {gen.extinctStrategies > 0 && (
              <div className="mt-2 text-xs text-red-600">
                🪦 {gen.extinctStrategies} strategies became extinct
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Fitness Progress Chart */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center">
            <TrendingUp className="w-5 h-5 mr-2" />
            Fitness Evolution Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <TimeSeriesChart
            data={formatChartData()}
            series={[
              {
                name: 'Best Fitness',
                dataKey: 'bestFitness',
                color: '#10B981'
              },
              {
                name: 'Average Fitness',
                dataKey: 'avgFitness',
                color: '#3B82F6'
              },
              {
                name: 'Population Diversity',
                dataKey: 'diversityScore',
                color: '#8B5CF6'
              }
            ]}
            xAxisDataKey="generation"
            showLegend={true}
          />
        </CardContent>
      </Card>
    </div>
  );

  const renderLineageView = () => (
    <div className="space-y-4">
      <div className="grid gap-4">
        {championLineage.map((champion) => (
          <Card key={champion.strategyId} className="border-l-4 border-l-yellow-500">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <Crown className="w-5 h-5 text-yellow-500" />
                  <h3 className="font-semibold text-foreground">{champion.name}</h3>
                  <Badge className="bg-yellow-100 text-yellow-800">
                    <Star className="w-3 h-3 mr-1" />
                    Champion
                  </Badge>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-green-600">
                    {champion.fitness.toFixed(3)}
                  </div>
                  <div className="text-sm text-gray-500">Peak: {champion.peakFitness.toFixed(3)}</div>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                <div>
                  <span className="text-xs text-gray-500">Generations Survived</span>
                  <div className="font-medium">{champion.generationsSurvived}</div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">First Appearance</span>
                  <div className="font-medium">Gen {champion.firstAppearance}</div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Last Seen</span>
                  <div className="font-medium">Gen {champion.lastAppearance}</div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Traits</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {champion.traits.slice(0, 2).map((trait) => (
                      <Badge key={trait} variant="outline" className="text-xs">
                        {trait}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              {/* Evolution Path Visualization */}
              <div className="mt-3">
                <span className="text-xs text-gray-500 mb-2 block">Evolution Path</span>
                <div className="flex items-center space-x-2">
                  {champion.evolutionPath.map((gen, index) => (
                    <React.Fragment key={gen}>
                      <div className="flex flex-col items-center">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                          index === champion.evolutionPath.length - 1
                            ? 'bg-yellow-500 text-white'
                            : 'bg-secondary text-secondary-foreground'
                        }`}>
                          {gen}
                        </div>
                        {index === champion.evolutionPath.length - 1 && (
                          <div className="text-xs text-yellow-600 mt-1">Current</div>
                        )}
                      </div>
                      {index < champion.evolutionPath.length - 1 && (
                        <div className="flex-1 h-0.5 bg-gray-300 min-w-4"></div>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  const renderDiversityView = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {liveGenerationsData.length > 0 ?
                liveGenerationsData[liveGenerationsData.length - 1]?.populationSize || 100 :
                '0'
              }
            </div>
            <div className="text-sm text-gray-600">Current Population</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {liveGenerationsData.length > 0 ?
                (liveGenerationsData[liveGenerationsData.length - 1]?.survivalRate * 100 || 0).toFixed(1) :
                '0.0'
              }%
            </div>
            <div className="text-sm text-gray-600">Survival Rate</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {liveGenerationsData.length > 0 ?
                (liveGenerationsData[liveGenerationsData.length - 1]?.diversityScore * 100 || 0).toFixed(1) :
                '0.0'
              }%
            </div>
            <div className="text-sm text-gray-600">Population Diversity</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Population Dynamics</CardTitle>
        </CardHeader>
        <CardContent>
          <TimeSeriesChart
            data={formatChartData()}
            series={[
              {
                name: 'Population Size',
                dataKey: 'populationSize',
                color: '#3B82F6'
              },
              {
                name: 'Survival Rate',
                dataKey: 'survivalRate',
                color: '#10B981'
              },
              {
                name: 'Diversity Score',
                dataKey: 'diversityScore',
                color: '#F59E0B'
              }
            ]}
            xAxisDataKey="generation"
            showLegend={true}
          />
        </CardContent>
      </Card>
    </div>
  );

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              {getFitnessTrendIcon()}
              <span className="ml-2">Evolution Lifecycle</span>
            </CardTitle>
            <p className="text-sm text-gray-600 mt-1">
              Track generational progress and strategy lineage
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className={`${
              getFitnessTrend() === 'improving' ? 'bg-green-100 text-green-800' :
              getFitnessTrend() === 'declining' ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {getFitnessTrend() === 'improving' ? 'Improving' :
               getFitnessTrend() === 'declining' ? 'Declining' : 'Stable'}
            </Badge>
            <Badge variant="outline">
              <Clock className="w-3 h-3 mr-1" />
              {generationLoading ? 'Loading...' : `${liveGenerationsData.length} Generations`}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as any)}>
          <TabsList className="mb-6">
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
            <TabsTrigger value="lineage">Champion Lineage</TabsTrigger>
            <TabsTrigger value="diversity">Population Dynamics</TabsTrigger>
          </TabsList>

          <TabsContent value="timeline">
            {renderTimelineView()}
          </TabsContent>

          <TabsContent value="lineage">
            {renderLineageView()}
          </TabsContent>

          <TabsContent value="diversity">
            {renderDiversityView()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default EvoLifecycleView;
