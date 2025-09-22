/**
 * ============================================
 * [CARD: RESEARCH & DISCOVERY HUB]
 * News analysis, fundamental research, strategy hypotheses, market discovery
 * ============================================
 */

import React, { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import {
  Search,
  Newspaper,
  TrendingUp,
  Brain,
  Target,
  Sparkles,
  Plus,
  Eye,
  Clock,
  AlertTriangle,
  CheckCircle,
  Zap,
  BarChart3,
  Globe,
  Filter,
  RefreshCw,
  Loader2
} from 'lucide-react';
import { showSuccessToast, showErrorToast } from '@/utils/toast.js';
import { contextApi, researchApi } from '@/services/api';

interface ResearchDiscoveryHubProps {
  onStartEvolutionWithSymbols?: (symbols: string[], config: any) => void;
  className?: string;
}

export const ResearchDiscoveryHub: React.FC<ResearchDiscoveryHubProps> = ({
  onStartEvolutionWithSymbols,
  className = ''
}) => {
  const queryClient = useQueryClient();
  const [selectedTab, setSelectedTab] = useState('sentiment');
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [researchFilter, setResearchFilter] = useState({
    sentimentThreshold: 0.05, // Lower threshold to show more symbols
    volumeThreshold: 1000000,
    sector: 'all',
    timeFrame: '24h'
  });

  // Real-time data connections - using available API functions
  const { data: newsData, isLoading: newsLoading } = useQuery({
    queryKey: ['news', researchFilter.timeFrame],
    queryFn: () => contextApi.getNews(25), // Use existing getNews function
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 15000,
  });

  // Fetch scanner candidates for sentiment-based symbols
  const { data: scannerData, isLoading: scannerLoading } = useQuery({
    queryKey: ['scanner-candidates'],
    queryFn: async () => {
      const response = await fetch('/api/scanner/candidates');
      if (!response.ok) throw new Error('Failed to fetch scanner candidates');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  // Fundamentals (real via /api/fundamentals)
  const { data: fundamentalsData, isLoading: fundamentalsLoading } = useQuery({
    queryKey: ['fundamentals'],
    queryFn: () => researchApi.getFundamentals().then(r => (r.success ? r.data : { items: [], asOf: new Date().toISOString() })),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  // Market discovery (real via /api/discovery/market)
  const { data: marketDiscovery, isLoading: discoveryLoading } = useQuery({
    queryKey: ['market-discovery'],
    queryFn: () => researchApi.getMarketDiscovery().then(r => (r.success ? r.data : { items: [], asOf: new Date().toISOString() })),
    refetchInterval: 45_000,
    staleTime: 20_000,
  });

  // Real news data (no mock fallback)
  const newsDiscoveries = Array.isArray((newsData as any)?.data) ? (newsData as any).data : [];

  // Real fundamentals data with fallback
  const fundamentalOpportunities = fundamentalsData?.items || [];
  /* fallback removed to ensure empty/neutral state until real data exists */

  // Fetch strategy hypotheses from API
  const { data: strategyHypothesesData } = useQuery({
    queryKey: ['evo', 'strategy-hypotheses'],
    queryFn: async () => {
      const response = await fetch('/api/evo/strategy-hypotheses');
      if (!response.ok) return [];
      return response.json();
    },
    refetchInterval: 60000,
    staleTime: 30000,
  });
  // Map API response to expected format
  const strategyHypotheses = (strategyHypothesesData || []).map((h: any) => ({
    id: h.id,
    name: h.id?.replace('_', ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
    description: h.hypothesis,
    strategyType: h.id,
    expectedSharpe: h.backtestResults?.sharpe || 0,
    expectedWinRate: h.backtestResults?.winRate || 0,
    riskLevel: h.confidence > 0.8 ? 'low' : h.confidence > 0.6 ? 'medium' : 'high',
    status: h.status,
    symbols: h.symbols || [] // Add empty array as default
  }));

  const handleSymbolToggle = (symbol: string) => {
    setSelectedCandidates(prev =>
      prev.includes(symbol)
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  const handleStartEvolution = () => {
    if (selectedCandidates.length === 0) {
      showErrorToast('Please select at least one symbol to test');
      return;
    }

    if (onStartEvolutionWithSymbols) {
      const config = {
        population_size: 100,
        generations: 50,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        symbols: selectedCandidates,
        optimization_metric: 'sharpe',
        sentiment_weight: researchFilter.sentimentThreshold,
        news_impact_weight: 0.2,
        intelligence_snowball: true,
        research_driven: true
      };

      onStartEvolutionWithSymbols(selectedCandidates, config);
      setSelectedCandidates([]);
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              <Search className="w-6 h-6 mr-2 text-blue-500" />
              Research & Discovery Hub
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              AI-powered research laboratory discovering trading opportunities
            </p>
          </div>
          {selectedCandidates.length > 0 && (
            <Button onClick={handleStartEvolution} className="flex items-center">
              <Zap className="w-4 h-4 mr-1" />
              Start Evolution ({selectedCandidates.length})
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={selectedTab} onValueChange={setSelectedTab}>
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="sentiment">Sentiment Leaders</TabsTrigger>
            <TabsTrigger value="news">News Intelligence</TabsTrigger>
            <TabsTrigger value="fundamentals">Fundamentals</TabsTrigger>
            <TabsTrigger value="strategies">Strategy Lab</TabsTrigger>
            <TabsTrigger value="discovery">Market Discovery</TabsTrigger>
          </TabsList>

          {/* Sentiment Leaders Tab */}
          <TabsContent value="sentiment" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                Bullish Sentiment Leaders
                {scannerLoading && <Loader2 className="w-4 h-4 ml-2 animate-spin text-blue-500" />}
              </h3>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => queryClient.invalidateQueries(['scanner-candidates'])}
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                Refresh
              </Button>
            </div>

            <div className="grid gap-4">
              {scannerData && Array.isArray(scannerData) ? (
                scannerData
                  .filter(candidate => candidate.confidence > researchFilter.sentimentThreshold)
                  .sort((a, b) => b.confidence - a.confidence)
                  .map((candidate) => (
                    <div 
                      key={candidate.symbol}
                      className={`border rounded-lg p-4 hover:shadow-md transition-all ${
                        selectedCandidates.includes(candidate.symbol) ? 'border-blue-500 bg-blue-50 dark:bg-blue-950' : ''
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h4 className="text-lg font-semibold">{candidate.symbol}</h4>
                            <Badge variant={candidate.confidence > 0.8 ? "default" : "secondary"}>
                              {(candidate.confidence * 100).toFixed(0)}% Confidence
                            </Badge>
                            <Badge variant="outline">
                              ${candidate.last?.toFixed(2) || 'N/A'}
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                            <span>Volume: {(candidate.volume / 1000000).toFixed(1)}M</span>
                            <span>Change: {candidate.change_pct?.toFixed(2) || 0}%</span>
                            <span>Spread: {candidate.spread_bps?.toFixed(1) || 'N/A'} bps</span>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant={selectedCandidates.includes(candidate.symbol) ? "default" : "outline"}
                          onClick={() => handleSymbolToggle(candidate.symbol)}
                        >
                          {selectedCandidates.includes(candidate.symbol) ? (
                            <>
                              <Eye className="w-3 h-3 mr-1" />
                              Selected
                            </>
                          ) : (
                            <>
                              <Plus className="w-3 h-3 mr-1" />
                              Add
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  ))
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No sentiment data available. Markets may be closed.
                </div>
              )}
            </div>
          </TabsContent>

          {/* News Intelligence Tab */}
          <TabsContent value="news" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center">
                <Newspaper className="w-5 h-5 mr-2" />
                News-Driven Discoveries
                {newsLoading && <Loader2 className="w-4 h-4 ml-2 animate-spin text-blue-500" />}
              </h3>
              <div className="flex items-center space-x-2">
                <Filter className="w-4 h-4" />
                <select
                  value={researchFilter.sentimentThreshold}
                  onChange={(e) => setResearchFilter(prev => ({ ...prev, sentimentThreshold: parseFloat(e.target.value) }))}
                  className="text-sm border rounded px-2 py-1"
                >
                  <option value="0.5">All Sentiment</option>
                  <option value="0.7">High Impact</option>
                  <option value="0.8">Very Positive</option>
                </select>
              </div>
            </div>

            <div className="space-y-4">
              {newsDiscoveries.map((news) => (
                <div key={news.id} className="border rounded-lg p-4 bg-card">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <Badge variant={news.sentiment > 0.8 ? "default" : news.sentiment > 0.5 ? "secondary" : "destructive"}>
                          {news.sentiment > 0.8 ? 'Very Bullish' : news.sentiment > 0.5 ? 'Bullish' : 'Bearish'}
                        </Badge>
                        <Badge variant="outline">{news.source}</Badge>
                        <Badge variant="outline" className={
                          news.impact === 'high' ? 'border-red-500 text-red-600' :
                          news.impact === 'medium' ? 'border-yellow-500 text-yellow-600' :
                          'border-green-500 text-green-600'
                        }>
                          {news.impact.toUpperCase()}
                        </Badge>
                      </div>
                      <h4 className="font-semibold mb-2">{news.title}</h4>
                      <p className="text-sm text-muted-foreground mb-3">{news.summary}</p>
                      <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                        <span>Sentiment: {(news.sentiment * 100).toFixed(0)}%</span>
                        <span>{new Date(news.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex flex-wrap gap-2">
                      {news.symbols.map((symbol) => (
                        <Button
                          key={symbol}
                          size="sm"
                          variant={selectedCandidates.includes(symbol) ? "default" : "outline"}
                          onClick={() => handleSymbolToggle(symbol)}
                          className="text-xs"
                        >
                          {selectedCandidates.includes(symbol) ? (
                            <>
                              <Eye className="w-3 h-3 mr-1" />
                              {symbol} ✓
                            </>
                          ) : (
                            <>
                              <Plus className="w-3 h-3 mr-1" />
                              {symbol}
                            </>
                          )}
                        </Button>
                      ))}
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const url = (news as any).url || (news as any).link;
                        if (url) window.open(url, '_blank', 'noopener,noreferrer');
                      }}
                      disabled={!(news as any).url && !(news as any).link}
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      Full Article
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Fundamentals Tab */}
          <TabsContent value="fundamentals" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Fundamental Analysis
              </h3>
              <Button size="sm" variant="outline">
                <RefreshCw className="w-4 h-4 mr-1" />
                Refresh Data
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {fundamentalsLoading && (
                <div className="text-sm text-muted-foreground">Loading fundamentals…</div>
              )}
              {!fundamentalsLoading && fundamentalOpportunities.length === 0 && (
                <div className="text-sm text-muted-foreground">No fundamentals available yet.</div>
              )}
              {fundamentalOpportunities.map((company) => (
                <div key={company.symbol} className="border rounded-lg p-4 bg-card">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold">{company.symbol}</h4>
                      <p className="text-sm text-muted-foreground">{company.company}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-600">
                        {company.researchScore}/10
                      </div>
                      <div className="text-xs text-muted-foreground">Research Score</div>
                    </div>
                  </div>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Market Cap:</span>
                      <span className="font-medium">{company.marketCap}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">P/E Ratio:</span>
                      <span className="font-medium">{company.peRatio}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Revenue Growth:</span>
                      <span className={`font-medium ${Number(company.revenueGrowth || 0) > 0.2 ? 'text-green-600' : Number(company.revenueGrowth || 0) > 0.1 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {(Number(company.revenueGrowth || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="mb-4">
                    <h5 className="text-sm font-medium mb-2">Key Catalysts:</h5>
                    <div className="flex flex-wrap gap-1">
                      {company.catalysts.slice(0, 2).map((catalyst) => (
                        <Badge key={catalyst} variant="outline" className="text-xs">
                          {catalyst}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <Button
                    size="sm"
                    variant={selectedCandidates.includes(company.symbol) ? "default" : "outline"}
                    onClick={() => handleSymbolToggle(company.symbol)}
                    className="w-full"
                  >
                    {selectedCandidates.includes(company.symbol) ? (
                      <>
                        <Eye className="w-3 h-3 mr-1" />
                        Selected for Evolution
                      </>
                    ) : (
                      <>
                        <Plus className="w-3 h-3 mr-1" />
                        Add to Research
                      </>
                    )}
                  </Button>
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Strategy Lab Tab */}
          <TabsContent value="strategies" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                Strategy Research Laboratory
              </h3>
              <Button size="sm" variant="outline">
                <Target className="w-4 h-4 mr-1" />
                Generate Hypothesis
              </Button>
            </div>

            <div className="space-y-4">
              {strategyHypotheses.map((hypothesis) => (
                <div key={hypothesis.id} className="border rounded-lg p-4 bg-card">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h4 className="font-semibold mb-2">{hypothesis.name}</h4>
                      <p className="text-sm text-muted-foreground mb-3">{hypothesis.description}</p>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                        <div>
                          <div className="text-xs text-muted-foreground">Strategy Type</div>
                          <div className="font-medium capitalize">{(hypothesis.strategyType || '').replace('_', ' ')}</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Expected Sharpe</div>
                          <div className="font-medium text-green-600">{hypothesis.expectedSharpe}</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Win Rate</div>
                          <div className="font-medium">{((hypothesis.expectedWinRate || 0) * 100).toFixed(0)}%</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Risk Level</div>
                          <Badge variant={
                            hypothesis.riskLevel === 'low' ? 'outline' :
                            hypothesis.riskLevel === 'medium' ? 'secondary' : 'destructive'
                          }>
                            {(hypothesis.riskLevel || 'unknown').toUpperCase()}
                          </Badge>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {hypothesis.symbols.map((symbol) => (
                          <Button
                            key={symbol}
                            size="sm"
                            variant={selectedCandidates.includes(symbol) ? "default" : "outline"}
                            onClick={() => handleSymbolToggle(symbol)}
                            className="text-xs"
                          >
                            {symbol}
                          </Button>
                        ))}
                      </div>
                    </div>

                    <div className="ml-4">
                      <Button size="sm" variant="outline">
                        <Target className="w-3 h-3 mr-1" />
                        Test Strategy
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Market Discovery Tab */}
          <TabsContent value="discovery" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center">
                <Globe className="w-5 h-5 mr-2" />
                Market Discovery Engine
              </h3>
              <div className="flex items-center space-x-2">
                <Button size="sm" variant="outline">
                  <Sparkles className="w-4 h-4 mr-1" />
                  Scan Universe
                </Button>
                <select
                  value={researchFilter.sector}
                  onChange={(e) => setResearchFilter(prev => ({ ...prev, sector: e.target.value }))}
                  className="text-sm border rounded px-2 py-1"
                >
                  <option value="all">All Sectors</option>
                  <option value="technology">Technology</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="finance">Finance</option>
                  <option value="energy">Energy</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Discovery Stats */}
              <div className="space-y-4">
                <h4 className="font-medium">Discovery Statistics</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="text-xl font-bold text-blue-600">4,247</div>
                    <div className="text-sm text-blue-800 dark:text-blue-200">Symbols Scanned</div>
                  </div>
                  <div className="bg-green-50 dark:bg-green-950/20 p-3 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="text-xl font-bold text-green-600">156</div>
                    <div className="text-sm text-green-800 dark:text-green-200">Opportunities Found</div>
                  </div>
                  <div className="bg-yellow-50 dark:bg-yellow-950/20 p-3 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <div className="text-xl font-bold text-yellow-600">23</div>
                    <div className="text-sm text-yellow-800 dark:text-yellow-200">High-Confidence</div>
                  </div>
                  <div className="bg-purple-50 dark:bg-purple-950/20 p-3 rounded-lg border border-purple-200 dark:border-purple-800">
                    <div className="text-xl font-bold text-purple-600">12</div>
                    <div className="text-sm text-purple-800 dark:text-purple-200">Under Research</div>
                  </div>
                </div>
              </div>

              {/* Sector Analysis */}
              <div className="space-y-4">
                <h4 className="font-medium">Sector Analysis</h4>
                <div className="space-y-3">
                  {[
                    { sector: 'Technology', opportunities: 45, total: 387, score: 8.7 },
                    { sector: 'Healthcare', opportunities: 23, total: 234, score: 7.9 },
                    { sector: 'Finance', opportunities: 18, total: 198, score: 7.2 },
                    { sector: 'Energy', opportunities: 12, total: 156, score: 6.8 }
                  ].map((sector) => (
                    <div key={sector.sector} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-medium">{sector.sector}</div>
                        <div className="text-sm text-muted-foreground">
                          {sector.opportunities} of {sector.total} opportunities
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-600">{sector.score}/10</div>
                        <div className="text-xs text-muted-foreground">Avg Score</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Emerging Opportunities */}
            <div className="space-y-4">
              <h4 className="font-medium">Emerging Opportunities</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {discoveryLoading && (
                  <div className="text-sm text-muted-foreground">Scanning market…</div>
                )}
                {!discoveryLoading && (marketDiscovery?.items?.length ?? 0) === 0 && (
                  <div className="text-sm text-muted-foreground">No discoveries yet.</div>
                )}
                {(marketDiscovery?.items || []).map((opportunity) => (
                  <div key={opportunity.symbol} className="border rounded-lg p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <div className="font-semibold">{opportunity.symbol}</div>
                        <div className="text-sm text-muted-foreground">{opportunity.company}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-green-600">{opportunity.score}</div>
                        <div className="text-xs text-muted-foreground">Score</div>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground mb-3">{opportunity.reason}</p>
                    <Button
                      size="sm"
                      variant={selectedCandidates.includes(opportunity.symbol) ? "default" : "outline"}
                      onClick={() => handleSymbolToggle(opportunity.symbol)}
                      className="w-full"
                    >
                      {selectedCandidates.includes(opportunity.symbol) ? 'Selected' : 'Research'}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Selection Summary */}
        {selectedCandidates.length > 0 && (
          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-yellow-900 dark:text-yellow-100">
                  Research Candidates Selected ({selectedCandidates.length})
                </h4>
                <p className="text-sm text-yellow-800 dark:text-yellow-200 mt-1">
                  Ready to start evolutionary testing: {selectedCandidates.join(', ')}
                </p>
              </div>
              <Button onClick={handleStartEvolution} className="bg-yellow-600 hover:bg-yellow-700">
                <Zap className="w-4 h-4 mr-1" />
                Launch Evolution
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
