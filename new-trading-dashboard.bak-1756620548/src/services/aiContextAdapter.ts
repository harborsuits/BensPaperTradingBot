/**
 * AI Context Adapter Service
 * 
 * This service is responsible for collecting and formatting all relevant trading context
 * from across the application to feed into the AI assistant. It ensures the AI has
 * complete awareness of the current trading environment including:
 * 
 * - Portfolio state and positions across all asset classes
 * - Market conditions, regimes, and real-time sentiment analysis
 * - Active strategies across all asset classes (stocks, options, crypto, forex)
 * - Trading opportunities, signals, and pattern discovery
 * - Recent trade decisions, their rationale, and execution details
 * - System health metrics, staging environment status, and risk management
 * - Data quality indicators and multi-source integration status
 * - External signals from TradingView, Alpaca, and Finnhub
 * - News analysis with portfolio impact assessments
 * - Backtesting and evolutionary testing results
 */

import axios from 'axios';

// Import the apiRequest function - redefining for simplicity
async function apiRequest<T>(config: any): Promise<T> {
  try {
    const response = await axios({
      baseURL: '/api',
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
      },
      ...config
    });
    return response.data;
  } catch (error) {
    console.error('API request error:', error);
    throw error;
  }
}
import { TradingContext } from './aiChatApi';

// Interface for complete orchestrator context
export interface OrchestratorContext {
  market: {
    regime: string;
    regimeConfidence: number;
    volatility: number;
    sentiment: number;
    sentimentFactors: {
      positive: { factor: string; impact: number }[];
      negative: { factor: string; impact: number }[];
    };
    keyLevels: Record<string, number[]>;
    majorIndices: Record<string, { price: number; change: number; volume: number }>;
    featureIndicators: {
      name: string;
      value: number;
      trend: 'up' | 'down' | 'neutral';
      relevance: number;
    }[];
    anomalies: {
      type: string;
      severity: number;
      description: string;
      timestamp: string;
    }[];
  };
  portfolio: {
    value: number;
    cashBalance: number;
    allocation: {
      stocks: number;
      options: number;
      crypto: number;
      forex: number;
      cash: number;
    };
    riskMetrics: {
      var: number; // Value at Risk
      maxDrawdown: number;
      beta: number;
      correlation: number;
      sharpe: number;
    };
    openPositions: {
      symbol: string;
      assetClass: 'stock' | 'option' | 'crypto' | 'forex';
      quantity: number;
      entryPrice: number;
      currentPrice: number;
      pnl: number;
      pnlPercent: number;
      strategy: string;
      entryTime: string;
      entryRationale: string;
      exitTarget: number;
      stopLoss: number;
    }[];
    recentTrades: {
      symbol: string;
      assetClass: 'stock' | 'option' | 'crypto' | 'forex';
      side: 'buy' | 'sell';
      quantity: number;
      price: number;
      timestamp: string;
      strategy: string;
      executionQuality: number;
      slippage: number;
      rationale: string;
      profitLoss: number;
    }[];
    performance: {
      daily: number;
      weekly: number;
      monthly: number;
      ytd: number;
      byAssetClass: {
        stocks: number;
        options: number;
        crypto: number;
        forex: number;
      };
      byStrategy: Record<string, number>;
    };
  };
  strategies: {
    active: string[];
    paused: string[];
    byAssetClass: {
      stocks: string[];
      options: string[];
      crypto: string[];
      forex: string[];
    };
    performance: Record<string, {
      winRate: number;
      profitFactor: number;
      expectancy: number;
      sharpe: number;
      trades: number;
      averageHoldTime: number;
      regimeCompatibility: Record<string, number>;
      bestTimeframes: string[];
    }>;
    activeSessions: {
      id: string;
      strategy: string;
      assetClass: 'stock' | 'option' | 'crypto' | 'forex';
      status: string;
      startTime: string;
      symbols: string[];
      performance: number;
    }[];
    rankingForCurrentMarket: {
      strategy: string;
      suitabilityScore: number;
      rationale: string;
    }[];
    patterns: {
      name: string;
      type: 'symbol-specific' | 'seasonal' | 'event-driven' | 'market-condition';
      confidence: number;
      description: string;
      applicableSymbols?: string[];
      applicableTimeframes?: string[];
      performanceHistory: {
        wins: number;
        losses: number;
        expectancy: number;
      };
    }[];
  };
  signals: {
    opportunities: {
      symbol: string;
      assetClass: 'stock' | 'option' | 'crypto' | 'forex';
      strategy: string;
      confidence: number;
      expectedReturn: number;
      timeframe: string;
      entryPrice: number;
      targetPrice: number;
      stopPrice: number;
      rationale: string;
      supportingFactors: string[];
      riskRewardRatio: number;
    }[];
    alerts: {
      id: string;
      type: string;
      message: string;
      severity: string;
      timestamp: string;
      relatedSymbols?: string[];
      actionable: boolean;
      recommendedAction?: string;
    }[];
    externalSignals: {
      source: 'TradingView' | 'Alpaca' | 'Finnhub';
      symbol: string;
      signalType: string;
      direction: 'buy' | 'sell' | 'neutral';
      confidence: number;
      timestamp: string;
      metadata: Record<string, any>;
    }[];
  };
  news: {
    marketImpact: {
      headline: string;
      summary: string;
      sentiment: number;
      marketImpact: number;
      portfolioImpact: number;
      timestamp: string;
      source: string;
      url?: string;
      relatedSymbols: string[];
      actionPlan?: string;
    }[];
    symbolSpecific: Record<string, {
      news: {
        headline: string;
        summary: string;
        sentiment: number;
        impact: number;
        timestamp: string;
        source: string;
      }[];
    }>;
  };
  backtesting: {
    recentResults: {
      id: string;
      strategy: string;
      symbols: string[];
      timeframe: string;
      startDate: string;
      endDate: string;
      returns: number;
      maxDrawdown: number;
      sharpe: number;
      winRate: number;
      tradesCount: number;
    }[];
    evolutionaryTesting: {
      activeRuns: {
        id: string;
        strategy: string;
        generation: number;
        population: number;
        bestFitness: number;
        progress: number;
      }[];
      recentFindings: {
        strategyVariant: string;
        improvement: number;
        keyParameters: Record<string, any>;
        performanceMetrics: Record<string, number>;
      }[];
    };
  };
  system: {
    health: {
      memoryUsage: number;
      cpuUsage: number;
      apiLatency: number;
      errorRate: number;
      lastRestart: string;
      uptime: number;
    };
    dataQuality: {
      marketDataFreshness: number;
      missingDataPoints: number;
      dataSources: string[];
      sourceStatus: Record<string, 'online' | 'degraded' | 'offline'>;
      dataLag: Record<string, number>;
    };
    staging: {
      active: boolean;
      mode: 'paper' | 'live' | 'mock';
      strategies: string[];
      healthMetrics: {
        memoryUsage: number;
        cpuUsage: number;
        latency: number;
        errorRate: number;
      };
      riskViolations: {
        type: string;
        count: number;
        lastOccurrence: string;
        severity: string;
      }[];
    };
    execution: {
      activeOrders: {
        id: string;
        symbol: string;
        type: string;
        status: string;
        placedAt: string;
      }[];
      advancedOrdersActive: {
        type: 'TWAP' | 'VWAP' | 'Iceberg' | 'Combo';
        symbol: string;
        progress: number;
        performance: number;
      }[];
      circuitBreakers: {
        active: boolean;
        triggers: {
          type: string;
          threshold: number;
          currentValue: number;
          status: 'ok' | 'warning' | 'triggered';
        }[];
      };
    };
    brokerIntelligence: {
      recommendations: {
        broker: string;
        score: number;
        rationale: string;
      }[];
      performance: Record<string, {
        reliability: number;
        latency: number;
        executionQuality: number;
        cost: number;
        overallScore: number;
      }>;
      failoverStatus: {
        active: boolean;
        lastSwitched: string;
        currentPrimary: string;
      };
    };
    aiModels: {
      active: string[];
      performance: Record<string, {
        accuracy: number;
        latency: number;
        usage: number;
      }>;
      routing: {
        defaultModel: string;
        specializedRoutes: {
          task: string;
          model: string;
          confidence: number;
        }[];
      };
    };
  };
}

/**
 * Fetches the complete trading context from the backend API
 * This provides the AI with full awareness of the entire system state
 */
export const getFullTradingContext = async (): Promise<OrchestratorContext> => {
  return apiRequest<OrchestratorContext>({
    url: '/ai/orchestrator/context',
    method: 'GET'
  });
};

/**
 * Enriches the basic trading context with comprehensive information
 * from across all trading systems and asset classes
 */
export const enrichTradingContext = async (
  basicContext: TradingContext
): Promise<TradingContext> => {
  // Attempt to get the full orchestrator context
  try {
    const fullContext = await getFullTradingContext();
    
    // Extract key strategy info across asset classes
    const activeStrategiesByClass = fullContext.strategies.byAssetClass || {};
    const allActiveStrategies = [
      ...(activeStrategiesByClass.stocks || []),
      ...(activeStrategiesByClass.options || []),
      ...(activeStrategiesByClass.crypto || []),
      ...(activeStrategiesByClass.forex || [])
    ];
    
    // Get top strategies for current market
    const topStrategies = fullContext.strategies.rankingForCurrentMarket
      ? fullContext.strategies.rankingForCurrentMarket
          .slice(0, 5)
          .map(s => `${s.strategy} (${s.suitabilityScore.toFixed(2)})`)
      : [];
    
    // Extract identified patterns
    const relevantPatterns = fullContext.strategies.patterns
      ? fullContext.strategies.patterns
          .filter(p => p.confidence > 0.7)
          .map(p => `${p.name} (${p.confidence.toFixed(2)})`)
      : [];
    
    // Get relevant news with portfolio impact
    const impactfulNews = fullContext.news?.marketImpact
      ? fullContext.news.marketImpact
          .filter(n => Math.abs(n.portfolioImpact) > 0.3)
          .slice(0, 3)
          .map(n => ({
            headline: n.headline,
            impact: n.portfolioImpact,
            action: n.actionPlan
          }))
      : [];
    
    // Get broker intelligence recommendations
    const brokerRecommendations = fullContext.system?.brokerIntelligence?.recommendations
      ? fullContext.system.brokerIntelligence.recommendations
          .slice(0, 2)
          .map(r => `${r.broker}: ${r.rationale}`)
      : [];
    
    // Get external signals
    const externalSignals = fullContext.signals?.externalSignals
      ? fullContext.signals.externalSignals
          .filter(s => new Date(s.timestamp).getTime() > Date.now() - 24*60*60*1000) // last 24 hours
          .slice(0, 3)
          .map(s => `${s.source} ${s.signalType} on ${s.symbol} (${s.direction})`)
      : [];
    
    // Get risk metrics
    const riskMetrics = fullContext.portfolio?.riskMetrics || {};
    
    // Get active executions
    const activeOrders = fullContext.system?.execution?.activeOrders?.length || 0;
    const advancedOrders = fullContext.system?.execution?.advancedOrdersActive || [];
    
    // System status
    const systemStatus = {
      healthy: (fullContext.system?.health?.errorRate || 0) < 0.05,
      stagingActive: fullContext.system?.staging?.active || false,
      dataQuality: Object.values(fullContext.system?.dataQuality?.sourceStatus || {}).every(s => s !== 'offline'),
      circuitBreakersActive: fullContext.system?.execution?.circuitBreakers?.active || false
    };
    
    // Enrich the context with comprehensive details
    return {
      ...basicContext,
      portfolioValue: fullContext.portfolio.value,
      openPositions: fullContext.portfolio.openPositions,
      market: {
        regime: fullContext.market.regime,
        regimeConfidence: fullContext.market.regimeConfidence,
        volatility: fullContext.market.volatility,
        sentiment: fullContext.market.sentiment,
        sentimentFactors: fullContext.market.sentimentFactors,
        anomalies: fullContext.market.anomalies
      },
      opportunities: fullContext.signals.opportunities,
      strategies: allActiveStrategies,
      topStrategies,
      systemHealth: fullContext.system.health,
      dataQuality: fullContext.system.dataQuality,
      alerts: fullContext.signals.alerts.slice(0, 5),
      tradingPatterns: relevantPatterns,
      news: impactfulNews,
      brokerRecommendations,
      externalSignals,
      riskMetrics,
      activeOrders,
      advancedOrders: advancedOrders.map(o => `${o.type} on ${o.symbol} (${o.progress}%)`),
      systemStatus,
      portfolioAllocation: fullContext.portfolio.allocation,
      performanceByAssetClass: fullContext.portfolio.performance.byAssetClass,
      backtestingResults: fullContext.backtesting?.recentResults?.slice(0, 3) || [],
      evolutionaryTesting: {
        activeRuns: fullContext.backtesting?.evolutionaryTesting?.activeRuns?.length || 0,
        recentFindings: fullContext.backtesting?.evolutionaryTesting?.recentFindings?.slice(0, 2) || []
      }
    };
  } catch (error) {
    console.error('Failed to fetch full orchestrator context:', error);
    // Return the basic context if we can't get the enriched data
    return basicContext;
  }
};

/**
 * Gets context specific to the current symbol being viewed
 */
export const getSymbolContext = async (symbol: string): Promise<any> => {
  return apiRequest({
    url: `/ai/context/symbol/${symbol}`,
    method: 'GET'
  });
};

/**
 * Creates a comprehensive prompt enhancement with all relevant trading context
 * to ensure the AI has complete awareness for decision support
 */
export const createContextPrompt = (context: TradingContext): string => {
  // Extract core information from context
  const { 
    currentTab, 
    symbol, 
    market, 
    openPositions = [], 
    opportunities = [],
    strategies = [],
    topStrategies = [],
    tradingPatterns = [],
    news = [],
    systemStatus = {},
    brokerRecommendations = [],
    externalSignals = [],
    activeOrders = 0,
    advancedOrders = [],
    portfolioAllocation = {},
    riskMetrics = {}
  } = context;
  
  // Extract market information
  const marketRegime = market?.regime;
  const marketSentiment = market?.sentiment;
  const marketVolatility = market?.volatility;
  const marketRegimeConfidence = (context as any).market?.regimeConfidence;
  const marketAnomalies = (context as any).market?.anomalies || [];
  
  // Build context-aware prompt enhancement
  let contextPrompt = `You are currently in the ${currentTab} view`;
  
  // Add symbol context
  if (symbol) {
    contextPrompt += ` looking at ${symbol}`;
  }
  
  // Add comprehensive market context
  if (marketRegime) {
    contextPrompt += `. The current market regime is ${marketRegime}`;
    if (marketRegimeConfidence) {
      contextPrompt += ` with ${(marketRegimeConfidence * 100).toFixed(0)}% confidence`;
    }
    if (marketSentiment !== undefined) {
      const sentimentDesc = marketSentiment > 0.6 ? 'positive' : 
                           marketSentiment < 0.4 ? 'negative' : 'neutral';
      contextPrompt += `, sentiment is ${sentimentDesc} (${(marketSentiment * 100).toFixed(0)}%)`;
    }
    if (marketVolatility !== undefined) {
      const volatilityDesc = marketVolatility > 0.7 ? 'high' : 
                            marketVolatility < 0.3 ? 'low' : 'moderate';
      contextPrompt += `, volatility is ${volatilityDesc}`;
    }
  }
  
  // Add portfolio context
  if (openPositions.length > 0) {
    const positionsByAssetClass = openPositions.reduce((acc: any, pos) => {
      const assetClass = pos.assetClass || 'unknown';
      acc[assetClass] = (acc[assetClass] || 0) + 1;
      return acc;
    }, {});
    
    const positionSummary = Object.entries(positionsByAssetClass)
      .map(([assetClass, count]) => `${count} ${assetClass}`)
      .join(', ');
    
    contextPrompt += `. There are ${openPositions.length} open positions (${positionSummary})`;
    
    // If looking at a specific symbol and have a position in it
    if (symbol) {
      const symbolPosition = openPositions.find(p => p.symbol === symbol);
      if (symbolPosition) {
        contextPrompt += ` including a position in ${symbol} with ${symbolPosition.pnlPercent.toFixed(2)}% P&L`;
        if (symbolPosition.entryRationale) {
          contextPrompt += ` (entry rationale: ${symbolPosition.entryRationale})`;
        }
      }
    }
  }
  
  // Add trading opportunities
  if (opportunities && opportunities.length > 0) {
    contextPrompt += `. There are ${opportunities.length} trading opportunities available`;
    
    // If looking at a specific symbol
    if (symbol) {
      const symbolOpportunity = opportunities.find(o => o.symbol === symbol);
      if (symbolOpportunity) {
        contextPrompt += ` including an opportunity for ${symbol} using the ${symbolOpportunity.strategy} strategy`;
        if (symbolOpportunity.confidence) {
          contextPrompt += ` (confidence: ${(symbolOpportunity.confidence * 100).toFixed(0)}%)`;
        }
        if (symbolOpportunity.rationale) {
          contextPrompt += ` based on: ${symbolOpportunity.rationale}`;
        }
      }
    }
  }
  
  // Add strategy context
  if (strategies && strategies.length > 0) {
    contextPrompt += `. Currently active strategies across all asset classes: ${strategies.join(', ')}`;
  }
  
  // Add top strategies for current market
  if (topStrategies && topStrategies.length > 0) {
    contextPrompt += `. Top strategies for current market conditions: ${topStrategies.join(', ')}`;
  }
  
  // Add trading patterns
  if (tradingPatterns && tradingPatterns.length > 0) {
    contextPrompt += `. Identified trading patterns: ${tradingPatterns.join(', ')}`;
  }
  
  // Add news context
  if (news && news.length > 0) {
    contextPrompt += `. Recent impactful news:`;
    news.forEach((item: any) => {
      const impact = item.impact > 0 ? 'positive' : 'negative';
      contextPrompt += ` ${item.headline} (${impact} impact: ${Math.abs(item.impact).toFixed(2)})`;
    });
  }
  
  // Add broker recommendations
  if (brokerRecommendations && brokerRecommendations.length > 0) {
    contextPrompt += `. Broker intelligence: ${brokerRecommendations.join('; ')}`;
  }
  
  // Add external signals
  if (externalSignals && externalSignals.length > 0) {
    contextPrompt += `. External signals: ${externalSignals.join(', ')}`;
  }
  
  // Add order execution context
  if (activeOrders && activeOrders > 0) {
    contextPrompt += `. ${activeOrders} orders are currently being executed`;
  }
  
  if (advancedOrders && advancedOrders.length > 0) {
    contextPrompt += `. Advanced orders in progress: ${advancedOrders.join(', ')}`;
  }
  
  // Add system status
  if (systemStatus && Object.keys(systemStatus).length > 0) {
    // Add warnings for any concerning system status
    const warnings = [];
    if (systemStatus.healthy === false) warnings.push('system health is degraded');
    if (systemStatus.dataQuality === false) warnings.push('data quality issues detected');
    if (systemStatus.circuitBreakersActive === true) warnings.push('circuit breakers active');
    
    if (warnings.length > 0) {
      contextPrompt += `. IMPORTANT SYSTEM WARNINGS: ${warnings.join(', ')}`;
    }
    
    if (systemStatus.stagingActive === true) {
      contextPrompt += `. System is running in staging mode for paper testing`;
    }
  }
  
  // Add risk metrics
  if (riskMetrics && Object.keys(riskMetrics).length > 0) {
    const { var: valueAtRisk, maxDrawdown, beta } = riskMetrics as any;
    if (valueAtRisk !== undefined) {
      contextPrompt += `. Portfolio VaR: ${(valueAtRisk * 100).toFixed(2)}%`;
    }
    if (maxDrawdown !== undefined) {
      contextPrompt += `. Max drawdown: ${(maxDrawdown * 100).toFixed(2)}%`;
    }
    if (beta !== undefined) {
      contextPrompt += `. Portfolio beta: ${beta.toFixed(2)}`;
    }
  }
  
  // Add market anomalies if present
  if (marketAnomalies && marketAnomalies.length > 0) {
    const highSeverityAnomalies = marketAnomalies.filter((a: any) => a.severity > 0.7);
    if (highSeverityAnomalies.length > 0) {
      contextPrompt += `. CRITICAL: ${highSeverityAnomalies.length} high-severity market anomalies detected`;
      highSeverityAnomalies.slice(0, 2).forEach((anomaly: any) => {
        contextPrompt += ` - ${anomaly.type}: ${anomaly.description}`;
      });
    }
  }
  
  return contextPrompt;
};

export default {
  getFullTradingContext,
  enrichTradingContext,
  getSymbolContext,
  createContextPrompt
};
