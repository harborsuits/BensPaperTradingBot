/**
 * ============================================
 * [CARD: PROMOTION PIPELINE]
 * Strategy promotion criteria, validation gates, deployment approval
 * ============================================
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import {
  ArrowUp,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp,
  Target,
  Award,
  AlertTriangle,
  Crown,
  Zap,
  RefreshCw,
  Loader2,
  Shield,
  Eye,
  FileText
} from 'lucide-react';
import ModeLabel from '@/components/ui/ModeLabel';
import { evoTesterApi, strategyApi } from '@/services/api';
import {
  useEvoPrecheck,
  useStageStrategy,
  useEvoWebSocketHandlers,
  PrecheckResult
} from '@/hooks/useEvoQueries';
import { usePromotionService } from '@/hooks/usePromotionService';

interface PromotionPipelineProps {
  className?: string;
}

const PromotionPipeline: React.FC<PromotionPipelineProps> = ({ className = '' }) => {
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [precheckResult, setPrecheckResult] = useState<PrecheckResult | null>(null);
  const [showPrecheck, setShowPrecheck] = useState(false);

  // Phase 3: Connect to EVO APIs
  useEvoWebSocketHandlers();

  // Get promotion candidates from evoTester history
  const { data: evoHistory, isLoading: evoLoading } = useQuery({
    queryKey: ['evoTester', 'history'],
    queryFn: () => evoTesterApi.getEvoHistory(),
    refetchInterval: 30000,
    staleTime: 15000,
  });

  // Precheck hook for selected strategy
  const { data: currentPrecheck, isLoading: precheckLoading } = useEvoPrecheck(
    selectedSession || '',
    selectedStrategy || ''
  );

  // Stage strategy mutation
  const stageMutation = useStageStrategy();

  // Update precheck result when it changes
  React.useEffect(() => {
    if (currentPrecheck) {
      setPrecheckResult(currentPrecheck);
    }
  }, [currentPrecheck]);

  // Legacy promotion service (will be phased out)
  const {
    pipelines,
    candidates,
    stats,
    promoteCandidate,
    getValidationResult,
    checkForPromotions
  } = usePromotionService();

  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);
  const [promoting, setPromoting] = useState<Set<string>>(new Set());

  // Phase 3: EVO Promotion Functions
  const handleEvoPrecheck = (sessionId: string, strategyId: string) => {
    setSelectedSession(sessionId);
    setSelectedStrategy(strategyId);
    setShowPrecheck(true);
  };

  const handleEvoStage = async (sessionId: string, strategyId: string, strategyRef: string) => {
    if (!precheckResult?.ok) return;

    try {
      await stageMutation.mutateAsync({
        sessionId,
        strategyRef,
        allocation: 3000, // Default allocation
        pool: 'EVO',
        ttlDays: 7
      });
    } catch (error) {
      console.error('Failed to stage EVO strategy:', error);
    }
  };

  // Legacy promotion function (will be phased out)
  const handlePromoteCandidate = async (candidateId: string, pipelineId: string) => {
    setPromoting(prev => new Set([...prev, candidateId]));

    try {
      const success = await promoteCandidate(candidateId, pipelineId);
      if (success) {
        console.log(`Successfully promoted candidate ${candidateId}`);
      }
    } catch (error) {
      console.error('Failed to promote candidate:', error);
    } finally {
      setPromoting(prev => {
        const newSet = new Set(prev);
        newSet.delete(candidateId);
        return newSet;
      });
    }
  };

  // Phase 3: EVO Promotion View
  const renderEvoPromotion = () => (
    <div className="space-y-6">
      {/* EVO Promotion Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <Shield className="w-5 h-5 mr-2 text-green-600" />
            EVO Strategy Promotion
            {evoLoading && <Loader2 className="w-4 h-4 ml-2 animate-spin text-green-600" />}
          </h3>
          <p className="text-sm text-gray-600">Promote winning strategies to paper EVO allocations</p>
        </div>
        <Button size="sm" variant="outline">
          <RefreshCw className="w-3 h-3 mr-1" />
          Refresh
        </Button>
      </div>

      {/* EVO Promotion Candidates */}
      <div className="space-y-4">
        {evoHistory?.experiments?.map((experiment: any) => {
          const sessionId = experiment.id;
          const topStrategy = experiment.topStrategies?.[0];

          if (!topStrategy) return null;

          return (
            <Card key={sessionId} className="border-l-4 border-l-green-500">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <Crown className="w-5 h-5 text-yellow-500" />
                    <div>
                      <h4 className="font-semibold text-foreground">{topStrategy.id}</h4>
                      <p className="text-sm text-gray-600">Session: {sessionId}</p>
                    </div>
                    <Badge className="bg-green-100 text-green-800 border-green-300">
                      WINNER
                    </Badge>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs">
                      EVO READY
                    </Badge>
                  </div>
                </div>

                {/* Strategy Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <span className="text-xs text-gray-500">Sharpe Ratio</span>
                    <div className="font-medium text-green-600">
                      {topStrategy.performance?.sharpeRatio?.toFixed(2) || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Max Drawdown</span>
                    <div className="font-medium text-red-600">
                      {topStrategy.performance?.maxDrawdown ? `${(topStrategy.performance.maxDrawdown * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Win Rate</span>
                    <div className="font-medium text-blue-600">
                      {topStrategy.performance?.winRate ? `${(topStrategy.performance.winRate * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Total Trades</span>
                    <div className="font-medium text-purple-600">
                      {topStrategy.performance?.tradeCount || 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Precheck Result */}
                {showPrecheck && selectedSession === sessionId && selectedStrategy === topStrategy.id && (
                  <Card className="mb-4 bg-blue-50 border-blue-200">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <Eye className="w-4 h-4 text-blue-600" />
                        <span className="text-sm font-medium text-blue-800">EVO Precheck Results</span>
                        {precheckLoading && <Loader2 className="w-4 h-4 animate-spin text-blue-600" />}
                      </div>

                      {precheckResult && (
                        <div className="space-y-2">
                          <div className="flex items-center space-x-2">
                            {precheckResult.ok ? (
                              <CheckCircle className="w-4 h-4 text-green-600" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-600" />
                            )}
                            <span className={`text-sm font-medium ${
                              precheckResult.ok ? 'text-green-800' : 'text-red-800'
                            }`}>
                              {precheckResult.ok ? 'PASS: Eligible for EVO' : 'FAIL: Not eligible'}
                            </span>
                          </div>

                          {!precheckResult.ok && precheckResult.failed && (
                            <div className="text-xs text-red-700 ml-6">
                              Failed checks: {precheckResult.failed.join(', ')}
                            </div>
                          )}

                          <div className="text-xs text-gray-600 ml-6">
                            Sharpe: {precheckResult.metrics.sharpe.toFixed(2)} |
                            Win Rate: {(precheckResult.metrics.win.toFixed(1))}% |
                            Trades: {precheckResult.metrics.trades}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}

                {/* Action Buttons */}
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-600">
                    Ready for EVO promotion with segregated capital allocation
                  </div>
                  <div className="flex space-x-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleEvoPrecheck(sessionId, topStrategy.id)}
                      disabled={precheckLoading}
                    >
                      <FileText className="w-3 h-3 mr-1" />
                      Precheck
                    </Button>
                    <Button
                      size="sm"
                      variant="default"
                      onClick={() => handleEvoStage(sessionId, topStrategy.id, topStrategy.id)}
                      disabled={!precheckResult?.ok || stageMutation.isPending}
                      className="bg-green-600 hover:bg-green-700"
                    >
                      {stageMutation.isPending ? (
                        <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      ) : (
                        <Shield className="w-3 h-3 mr-1" />
                      )}
                      Stage for EVO
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        }) || []}

        {(!evoHistory?.experiments || evoHistory.experiments.length === 0) && !evoLoading && (
          <div className="text-center py-8 text-gray-500">
            <Crown className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Promotion Candidates</h3>
            <p className="text-gray-600">Complete EvoTester sessions to see strategies ready for EVO promotion</p>
          </div>
        )}
      </div>

      {/* EVO Promotion Stats */}
      {evoHistory?.experiments && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-green-600 mb-1">
                {evoHistory.experiments.filter((exp: any) => exp.topStrategies?.[0]).length}
              </div>
              <div className="text-sm text-gray-600">Ready for EVO</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-blue-600 mb-1">
                {evoHistory.experiments.length}
              </div>
              <div className="text-sm text-gray-600">Total Sessions</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-purple-600 mb-1">
                {Math.round(evoHistory.experiments.reduce((acc: number, exp: any) =>
                  acc + (exp.topStrategies?.[0]?.performance?.sharpeRatio || 0), 0) /
                  evoHistory.experiments.filter((exp: any) => exp.topStrategies?.[0]).length * 100) / 100 || 0}
              </div>
              <div className="text-sm text-gray-600">Avg Sharpe</div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );

  const renderPipelineOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-600 mb-1">
            {stats.totalCandidates}
          </div>
          <div className="text-sm text-foreground">Candidates</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-green-600 mb-1">
            {stats.totalPromoted}
          </div>
          <div className="text-sm text-foreground">Promoted</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-purple-600 mb-1">
            {(stats.successRate * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Success Rate</div>
        </CardContent>
      </Card>
    </div>
  );

  const renderPipelineDetails = () => (
    <div className="space-y-4">
      {pipelines.map((pipeline) => (
        <Card key={pipeline.id} className="border-l-4 border-l-blue-500">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-foreground">{pipeline.name}</h3>
                <p className="text-sm text-foreground">
                  {pipeline.criteria.minGenerations}+ generations • Fitness ≥ {pipeline.criteria.minFitness}
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant={pipeline.active ? "default" : "secondary"}>
                  {pipeline.active ? 'Active' : 'Inactive'}
                </Badge>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setSelectedPipeline(
                    selectedPipeline === pipeline.id ? null : pipeline.id
                  )}
                >
                  {selectedPipeline === pipeline.id ? 'Hide' : 'View'} Details
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="text-center">
                <div className="text-lg font-semibold text-blue-600">
                  {pipeline.candidates.length}
                </div>
                <div className="text-xs text-gray-600">Candidates</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600">
                  {pipeline.promoted.length}
                </div>
                <div className="text-xs text-gray-600">Promoted</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-red-600">
                  {pipeline.rejected.length}
                </div>
                <div className="text-xs text-gray-600">Rejected</div>
              </div>
            </div>

            {selectedPipeline === pipeline.id && (
              <div className="mt-4 pt-4 border-t">
                {/* Candidates Section */}
                {pipeline.candidates.length > 0 && (
                  <div className="mb-4">
                    <h4 className="font-medium text-foreground mb-3">Ready for Promotion</h4>
                    <div className="space-y-3">
                      {pipeline.candidates.map((candidate) => {
                        const isPromoting = promoting.has(candidate.id);
                        const validationResult = getValidationResult(candidate.id);

                        return (
                          <div key={candidate.id} className="border rounded-lg p-3 bg-gray-50">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-3">
                                <Crown className="w-5 h-5 text-yellow-500" />
                                <div>
                                  <h5 className="font-medium text-foreground">{candidate.name}</h5>
                                  <p className="text-sm text-foreground">
                                    Fitness: {candidate.fitness.toFixed(3)} • Gen: {candidate.generation}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-2">
                                <ModeLabel mode="research" size="sm" />
                                <Badge variant="outline" className="text-xs">
                                  {candidate.metadata.riskLevel.toUpperCase()}
                                </Badge>
                                <Button
                                  size="sm"
                                  onClick={() => handlePromoteCandidate(candidate.id, pipeline.id)}
                                  disabled={isPromoting}
                                >
                                  {isPromoting ? (
                                    <>
                                      <Zap className="w-3 h-3 mr-1 animate-spin" />
                                      Promoting...
                                    </>
                                  ) : (
                                    <>
                                      <ArrowUp className="w-3 h-3 mr-1" />
                                      Promote
                                    </>
                                  )}
                                </Button>
                              </div>
                            </div>

                            <div className="grid grid-cols-4 gap-4 text-sm">
                              <div>
                                <span className="text-gray-500">Win Rate:</span>
                                <div className="font-medium text-green-600">
                                  {(candidate.performance.winRate * 100).toFixed(1)}%
                                </div>
                              </div>
                              <div>
                                <span className="text-gray-500">Profit Factor:</span>
                                <div className="font-medium text-blue-600">
                                  {candidate.performance.profitFactor.toFixed(2)}
                                </div>
                              </div>
                              <div>
                                <span className="text-gray-500">Max DD:</span>
                                <div className="font-medium text-red-600">
                                  {(candidate.performance.maxDrawdown * 100).toFixed(1)}%
                                </div>
                              </div>
                              <div>
                                <span className="text-gray-500">Sharpe:</span>
                                <div className="font-medium text-purple-600">
                                  {candidate.performance.sharpeRatio.toFixed(2)}
                                </div>
                              </div>
                            </div>

                            {validationResult && (
                              <div className="mt-3 p-2 bg-card text-foreground rounded border border-border">
                                <div className="flex items-center space-x-2 mb-2">
                                  {validationResult.passed ? (
                                    <CheckCircle className="w-4 h-4 text-green-500" />
                                  ) : (
                                    <XCircle className="w-4 h-4 text-red-500" />
                                  )}
                                  <span className="text-sm font-medium">
                                    Validation {validationResult.passed ? 'Passed' : 'Failed'}
                                  </span>
                                  <Badge variant="outline" className="text-xs">
                                    Score: {(validationResult.score * 100).toFixed(1)}%
                                  </Badge>
                                </div>
                                <div className="text-xs text-gray-600">
                                  {validationResult.feedback.map((item, index) => (
                                    <div key={index}>{item}</div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Promoted Section */}
                {pipeline.promoted.length > 0 && (
                  <div className="mb-4">
                    <h4 className="font-medium text-foreground mb-3">Successfully Promoted</h4>
                    <div className="space-y-2">
                      {pipeline.promoted.map((candidate) => (
                        <div key={candidate.id} className="flex items-center justify-between p-2 bg-green-50 rounded">
                          <div className="flex items-center space-x-2">
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            <span className="font-medium text-green-800">{candidate.name}</span>
                            <ModeLabel mode="competition" size="sm" />
                          </div>
                          <Badge className="bg-green-100 text-green-800">
                            Fitness: {candidate.fitness.toFixed(3)}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Rejected Section */}
                {pipeline.rejected.length > 0 && (
                  <div>
                    <h4 className="font-medium text-foreground mb-3">Rejected Candidates</h4>
                    <div className="space-y-2">
                      {pipeline.rejected.map((candidate) => (
                        <div key={candidate.id} className="flex items-center justify-between p-2 bg-red-50 rounded">
                          <div className="flex items-center space-x-2">
                            <XCircle className="w-4 h-4 text-red-500" />
                            <span className="font-medium text-red-800">{candidate.name}</span>
                            <ModeLabel mode="research" size="sm" />
                          </div>
                          <span className="text-sm text-red-600">
                            Failed validation requirements
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderPromotionCriteria = () => (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Promotion Criteria Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Pipeline</th>
                  <th className="text-center py-2">Min Generations</th>
                  <th className="text-center py-2">Min Fitness</th>
                  <th className="text-center py-2">Min Win Rate</th>
                  <th className="text-center py-2">Max Drawdown</th>
                  <th className="text-center py-2">Validation Days</th>
                  <th className="text-center py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {pipelines.map((pipeline) => (
                  <tr key={pipeline.id} className="border-b">
                    <td className="py-2 font-medium">{pipeline.name}</td>
                    <td className="py-2 text-center">{pipeline.criteria.minGenerations}</td>
                    <td className="py-2 text-center">{pipeline.criteria.minFitness}</td>
                    <td className="py-2 text-center">{(pipeline.criteria.minWinRate * 100).toFixed(0)}%</td>
                    <td className="py-2 text-center">{(pipeline.criteria.maxDrawdown * 100).toFixed(0)}%</td>
                    <td className="py-2 text-center">{pipeline.criteria.validationPeriod}</td>
                    <td className="py-2 text-center">
                      <Badge variant={pipeline.active ? "default" : "secondary"}>
                        {pipeline.active ? 'Active' : 'Inactive'}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
              <ArrowUp className="w-5 h-5 mr-2 text-blue-500" />
              Promotion Pipeline
            </CardTitle>
            <p className="text-sm text-gray-600 mt-1">
              Automatic promotion from research to competition
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button size="sm" variant="outline" onClick={checkForPromotions}>
              <Zap className="w-4 h-4 mr-1" />
              Check Promotions
            </Button>
            <Badge variant="outline">
              <Award className="w-3 h-3 mr-1" />
              {stats.activePipelines} Active
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {renderPipelineOverview()}

        <Tabs defaultValue="evo">
          <TabsList className="mb-6">
            <TabsTrigger value="evo">EVO Promotion</TabsTrigger>
            <TabsTrigger value="pipelines">Promotion Pipelines</TabsTrigger>
            <TabsTrigger value="criteria">Promotion Criteria</TabsTrigger>
          </TabsList>

          <TabsContent value="evo">
            {renderEvoPromotion()}
          </TabsContent>

          <TabsContent value="pipelines">
            {renderPipelineDetails()}
          </TabsContent>

          <TabsContent value="criteria">
            {renderPromotionCriteria()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default PromotionPipeline;
