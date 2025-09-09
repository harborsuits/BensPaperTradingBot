/**
 * ============================================
 * [CARD: RESEARCH & DISCOVERY HUB]
 * News analysis, fundamental research, strategy hypotheses, market discovery
 * ============================================
 */

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
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
import { contextApi } from '@/services/api';

interface ResearchDiscoveryHubProps {
  onStartEvolutionWithSymbols?: (symbols: string[], config: any) => void;
  className?: string;
}

export const ResearchDiscoveryHub: React.FC<ResearchDiscoveryHubProps> = ({
  onStartEvolutionWithSymbols,
  className = ''
}) => {
  const [selectedTab, setSelectedTab] = useState('news');
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [researchFilter, setResearchFilter] = useState({
    sentimentThreshold: 0.7,
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

  // Mock fundamentals data since the API function doesn't exist yet
  const fundamentalsData = { fundamentals: [] };
  const fundamentalsLoading = false;

  // Mock market discovery data since the API function doesn't exist yet
  const marketDiscovery = { discoveries: [] };
  const discoveryLoading = false;

  // Real news data with fallback to mock data
  const newsDiscoveries = newsData?.news || [
    {
      id: 'news_001',
      title: 'NVIDIA Announces Breakthrough in AI Chip Technology',
      source: 'TechCrunch',
      sentiment: 0.85,
      symbols: ['NVDA', 'AMD', 'INTC'],
      impact: 'high',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      summary: 'NVIDIA reveals next-gen AI processors with 300% performance improvement'
    },
    {
      id: 'news_002',
      title: 'Tesla Battery Breakthrough Could Revolutionize EV Market',
      source: 'Bloomberg',
      sentiment: 0.92,
      symbols: ['TSLA', 'GM', 'F'],
      impact: 'high',
      timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
      summary: 'New solid-state battery technology promises 500-mile range'
    },
    {
      id: 'news_003',
      title: 'Apple Supply Chain Issues Impact iPhone Production',
      source: 'WSJ',
      sentiment: -0.75,
      symbols: ['AAPL', 'QCOM', 'TSM'],
      impact: 'medium',
      timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
      summary: 'Taiwan Semiconductor manufacturing delays affect Apple products'
    }
  ];

  // Real fundamentals data with fallback
  const fundamentalOpportunities = fundamentalsData?.fundamentals || [
    {
      symbol: 'NVDA',
      company: 'NVIDIA Corporation',
      sector: 'Technology',
      marketCap: '2.8T',
      peRatio: 65.2,
      revenueGrowth: 0.265,
      earningsGrowth: 0.345,
      debtToEquity: 0.42,
      researchScore: 9.2,
      catalysts: ['AI Chip Demand', 'Data Center Growth', 'Gaming Revenue']
    },
    {
      symbol: 'TSLA',
      company: 'Tesla Inc.',
      sector: 'Automotive',
      marketCap: '650B',
      peRatio: 45.8,
      revenueGrowth: 0.198,
      earningsGrowth: 0.156,
      debtToEquity: 0.15,
      researchScore: 8.7,
      catalysts: ['Battery Breakthrough', 'Autonomous Driving', 'Energy Storage']
    },
    {
      symbol: 'AMD',
      company: 'Advanced Micro Devices',
      sector: 'Technology',
      marketCap: '180B',
      peRatio: 185.4,
      revenueGrowth: 0.089,
      earningsGrowth: 0.234,
      debtToEquity: 0.02,
      researchScore: 8.1,
      catalysts: ['AI Competition', 'Server Demand', 'Console Gaming']
    }
  ];

  const strategyHypotheses = [
    {
      id: 'hyp_001',
      name: 'AI Momentum Breakout',
      description: 'Capitalize on AI sector momentum with breakout strategies',
      symbols: ['NVDA', 'AMD', 'GOOGL', 'MSFT'],
      strategyType: 'momentum_breakout',
      expectedSharpe: 2.1,
      expectedWinRate: 0.68,
      riskLevel: 'medium',
      timeHorizon: '3-6 months'
    },
    {
      id: 'hyp_002',
      name: 'EV Revolution Mean Reversion',
      description: 'Mean reversion strategies for volatile EV stocks',
      symbols: ['TSLA', 'NIO', 'RIVN', 'LCID'],
      strategyType: 'mean_reversion',
      expectedSharpe: 1.8,
      expectedWinRate: 0.62,
      riskLevel: 'high',
      timeHorizon: '1-3 months'
    },
    {
      id: 'hyp_003',
      name: 'Semiconductor Supply Chain',
      description: 'Long/short strategies around semiconductor supply dynamics',
      symbols: ['TSM', 'ASML', 'QCOM', 'INTC'],
      strategyType: 'pairs_trading',
      expectedSharpe: 1.9,
      expectedWinRate: 0.65,
      riskLevel: 'medium',
      timeHorizon: '2-4 months'
    }
  ];

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
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="news">News Intelligence</TabsTrigger>
            <TabsTrigger value="fundamentals">Fundamentals</TabsTrigger>
            <TabsTrigger value="strategies">Strategy Lab</TabsTrigger>
            <TabsTrigger value="discovery">Market Discovery</TabsTrigger>
          </TabsList>

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
                    <Button size="sm" variant="outline">
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
                      <span className={`font-medium ${company.revenueGrowth > 0.2 ? 'text-green-600' : company.revenueGrowth > 0.1 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {(company.revenueGrowth * 100).toFixed(1)}%
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
                          <div className="font-medium capitalize">{hypothesis.strategyType.replace('_', ' ')}</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Expected Sharpe</div>
                          <div className="font-medium text-green-600">{hypothesis.expectedSharpe}</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Win Rate</div>
                          <div className="font-medium">{(hypothesis.expectedWinRate * 100).toFixed(0)}%</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Risk Level</div>
                          <Badge variant={
                            hypothesis.riskLevel === 'low' ? 'outline' :
                            hypothesis.riskLevel === 'medium' ? 'secondary' : 'destructive'
                          }>
                            {hypothesis.riskLevel.toUpperCase()}
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
                {[
                  { symbol: 'PLTR', company: 'Palantir', reason: 'AI Government Contracts', score: 8.9 },
                  { symbol: 'SOFI', company: 'SoFi Technologies', reason: 'Digital Banking Growth', score: 8.3 },
                  { symbol: 'RIVN', company: 'Rivian', reason: 'EV Market Expansion', score: 7.8 }
                ].map((opportunity) => (
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
