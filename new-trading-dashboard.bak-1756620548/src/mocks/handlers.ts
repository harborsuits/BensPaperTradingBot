import { http, HttpResponse } from 'msw';

export const handlers = [
  // Universe / Watchlists
  http.get('/api/watchlists', () => {
    const lists = {
      top_etfs: ['SPY','QQQ','DIA','IWM'],
      faang: ['AAPL','MSFT','GOOGL','META','AMZN','NFLX'],
      semis: ['NVDA','AMD','AVGO','TSM','MU','SMCI'],
      my: ['AAPL','TSLA','AMD','SMCI']
    };
    const items = Object.entries(lists).map(([id, syms]) => ({ id, count: (syms as string[]).length }));
    return HttpResponse.json({ items });
  }),
  http.get('/api/universe', () => HttpResponse.json({ id: 'faang', symbols: ['AAPL','MSFT','GOOGL','META','AMZN','NFLX'] })),
  http.post('/api/universe', async ({ request }) => {
    const body = await request.json().catch(() => ({}));
    return HttpResponse.json({ ok: true, id: (body as any)?.id ?? 'faang' });
  }),
  // Generic context aggregate used by Dashboard
  http.get('/api/context', () =>
    HttpResponse.json({
      asOf: new Date().toISOString(),
      regime: { label: 'neutral', confidence: 0.58 },
      sentiment: { score: 0.23, scale: [-1, 1], asOf: new Date().toISOString() },
      features: {
        momentum: { value: 0.65, change: 2.1 },
        volatility: { value: 0.48, change: -1.2 },
        breadth: { value: 0.72, change: 3.5 },
        liquidity: { value: 0.52, change: 0.3 },
        sentiment: { value: 0.38, change: -0.8 },
        fundamentals: { value: 0.61, change: 0.0 },
        technicals: { value: 0.58, change: 1.7 },
        correlation: { value: 0.42, change: -0.5 }
      },
      news: []
    })
  ),
  // Context API mocks
  http.get('/api/context/regime', () => 
    HttpResponse.json({ 
      regime: 'neutral', 
      confidence: 0.5,
      since: new Date().toISOString(),
      description: 'Market is showing neutral patterns with balanced buying and selling pressure.'
    })
  ),
  
  http.get('/api/context/features', () => 
    HttpResponse.json({ 
      features: [] 
    })
  ),
  
  http.get('/api/context/sentiment', () => 
    HttpResponse.json({ 
      overall_score: 0.2,
      market_sentiment: 'neutral',
      timestamp: new Date().toISOString(),
      source: 'Mock Data',
      positive_factors: ['Stable economic indicators', 'Consistent trading volumes'],
      negative_factors: ['Geopolitical uncertainties', 'Mixed earnings reports']
    })
  ),
  
  http.get('/api/context/sentiment/history', () =>
    HttpResponse.json([
      {
        timestamp: new Date(Date.now() - 86400000 * 6).toISOString(),
        score: -0.2,
        sentiment: 'negative'
      },
      {
        timestamp: new Date(Date.now() - 86400000 * 5).toISOString(),
        score: -0.1,
        sentiment: 'negative'
      },
      {
        timestamp: new Date(Date.now() - 86400000 * 4).toISOString(),
        score: 0,
        sentiment: 'neutral'
      },
      {
        timestamp: new Date(Date.now() - 86400000 * 3).toISOString(),
        score: 0.1,
        sentiment: 'neutral'
      },
      {
        timestamp: new Date(Date.now() - 86400000 * 2).toISOString(),
        score: 0.2,
        sentiment: 'positive'
      },
      {
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        score: 0.3,
        sentiment: 'positive'
      },
      {
        timestamp: new Date().toISOString(),
        score: 0.2,
        sentiment: 'positive'
      }
    ])
  ),
  
  http.get('/api/context/sentiment/anomalies', () =>
    HttpResponse.json([])
  ),
  
  http.get('/api/context/news', ({ request }) => {
    const url = new URL(request.url);
    const limit = Number(url.searchParams.get('limit') || 10);
    const items = Array.from({ length: limit }, (_, i) => ({
      id: `news-${i + 1}`,
      headline: `Mock headline ${i + 1}`,
      summary: `Summary for mock headline ${i + 1}`,
      url: `https://example.com/news/${i + 1}`,
      source: 'MSW Times',
      published_at: new Date(Date.now() - i * 3600000).toISOString(),
      sentiment_score: ((i % 5) - 2) / 2,
      impact: (['high','medium','low'] as const)[i % 3],
      categories: ['markets'],
      symbols: i % 2 === 0 ? ['SPY', 'QQQ'] : ['AAPL'],
    }));
    return HttpResponse.json(items);
  }),

  // Decisions endpoints (provide fields used by TradeDecisionsPage)
  http.get('/api/decisions', ({ request }) => {
    const now = new Date();
    const iso = (d: Date) => d.toISOString();
    const items = [
      { id: 'dec_1', timestamp: iso(new Date(now.getTime() - 60_000)), created_at: iso(new Date(now.getTime() - 60_000)), symbol: 'SPY', strategy_id: 's1', strategy_name: 'ETF Trend', action: 'buy', decided: 'executed', score: 0.71, reason: 'trend + breadth ok' },
      { id: 'dec_2', timestamp: iso(new Date(now.getTime() - 30_000)), created_at: iso(new Date(now.getTime() - 30_000)), symbol: 'BTC-USD', strategy_id: 's2', strategy_name: 'Crypto Burst', action: 'buy', decided: 'skipped', score: 0.64, reason: 'news burst' },
      { id: 'dec_3', timestamp: iso(now), created_at: iso(now), symbol: 'AAPL', strategy_id: 's3', strategy_name: 'Earnings Drift', action: 'sell', decided: 'executed', score: 0.58, reason: 'post-earnings drift' }
    ];
    return HttpResponse.json(items);
  }),
  http.get('/api/decisions/latest', () => {
    const iso = (d: Date) => d.toISOString();
    const now = new Date();
    return HttpResponse.json([
      { id: 'dec_latest', timestamp: iso(now), created_at: iso(now), symbol: 'QQQ', strategy_id: 's4', strategy_name: 'Mean Revert', action: 'sell', decided: 'skipped', score: 0.52, reason: 'overbought' }
    ]);
  }),

  // Safety status (shape matches SafetyControls expectations)
  http.get(/\/_?api\/safety\/status$/, () =>
    HttpResponse.json({
      asOf: new Date().toISOString(),
      tradingMode: 'paper',
      emergencyStopActive: false,
      circuitBreakers: {
        active: false,
        reason: null,
        triggeredAt: null,
        maxDailyLoss: 0,
        currentDailyLoss: 0,
        maxTradesPerDay: 0,
        currentTradeCount: 0,
      },
      cooldowns: {
        active: false,
        endsAt: null,
        remainingSeconds: 0,
        reason: null,
      },
    })
  ),

  // Portfolio & Trades (minimal shapes)
  http.get('/api/portfolio', () =>
    HttpResponse.json({ summary: { total_equity: 100000, cash_balance: 20000, daily_pl: 250, daily_pl_percent: 0.25 }, positions: [] })
  ),
  http.get('/api/trades', () => HttpResponse.json([])),

  // Prometheus-style metrics text
  http.get('/metrics', () =>
    new HttpResponse([
      '# HELP app_up 1=up',
      '# TYPE app_up gauge',
      'app_up 1',
      '# HELP trading_health_ok 1=healthy',
      '# TYPE trading_health_ok gauge',
      'trading_health_ok{source="alpaca"} 1',
      'trading_health_ok{source="websocket"} 1',
      'trading_health_ok{source="news"} 1',
    ].join('\n'), { headers: { 'Content-Type': 'text/plain; version=0.0.4' } })
  ),

  // Robust alerts matcher (handles query strings)
  http.get(/\/api\/alerts$/, ({ request }) => {
    const url = new URL(request.url);
    const limit = Number(url.searchParams.get('limit') ?? 20);
    const severity = Number(url.searchParams.get('severity') ?? 3);
    const items = Array.from({ length: Math.min(limit, 5) }).map((_, i) => ({
      id: `alert_${i}`,
      severity,
      message: severity >= 5 ? 'Risk gate engaged: max position size enforced' : 'Heads up: elevated volatility',
      createdAt: new Date().toISOString(),
    }));
    return HttpResponse.json(items);
  }),
  
  // Strategy API mocks
  http.get('/api/strategies', () => 
    HttpResponse.json({ 
      items: [
        {
          id: '1',
          name: 'Adaptive Momentum',
          status: 'active',
          asset_class: 'equities',
          description: 'Momentum strategy with volatility adaptation',
          priority_score: 0.85,
          performance: {
            win_rate: 0.68,
            sharpe_ratio: 1.42,
            max_drawdown: 0.12,
            trades_count: 124
          }
        },
        {
          id: '2',
          name: 'Mean Reversion',
          status: 'idle',
          asset_class: 'equities',
          description: 'Reversal strategy for overbought/oversold conditions',
          priority_score: 0.65,
          performance: {
            win_rate: 0.62,
            sharpe_ratio: 1.28,
            max_drawdown: 0.15,
            trades_count: 98
          }
        }
      ]
    })
  ),
  // Active strategies endpoint used by ActiveStrategiesCard
  http.get('/api/strategies/active', () =>
    HttpResponse.json([
      { id: 's1', name: 'Adaptive Momentum', signal: 'BUY', confidence: 0.72, positions: 3, pnlDay: 185.34, health: 'ok', asOf: new Date().toISOString() },
      { id: 's2', name: 'Mean Reversion', signal: 'SELL', confidence: 0.58, positions: 2, pnlDay: -42.11, health: 'degraded', asOf: new Date().toISOString() },
    ])
  ),
  
  // EvoTester mocks
  http.get('/api/evotester/history', () =>
    HttpResponse.json([
      {
        id: 'evo-1',
        date: new Date().toISOString(),
        bestFitness: 0.78,
        status: 'completed',
        generations: 50,
        elapsed: '1h 23m'
      },
      {
        id: 'evo-2',
        date: new Date(Date.now() - 86400000 * 2).toISOString(),
        bestFitness: 0.72,
        status: 'completed',
        generations: 42,
        elapsed: '58m'
      }
    ])
  ),
  
  // WebSocket ping-pong
  http.post('/api/ws/ping', () => HttpResponse.json({ type: 'pong', timestamp: Date.now() })),
  
  // Portfolio mocks
  http.get('/api/portfolio', () =>
    HttpResponse.json({ 
      summary: { 
        total_equity: 10000, 
        cash_balance: 5000, 
        daily_pl: 250, 
        daily_pl_percent: 2.5 
      },
      positions: []
    })
  ),
  
  // Portfolio history for paper/live accounts
  http.get(/\/api\/portfolio\/(paper|live)\/history$/, ({ request }) => {
    const url = new URL(request.url);
    const days = Math.max(1, Math.min(365, Number(url.searchParams.get('days') || 30)));
    const now = Date.now();
    const points = Array.from({ length: days }).map((_, i) => {
      const t = new Date(now - (days - 1 - i) * 24 * 3600 * 1000);
      const base = 10000;
      const drift = i * 7;
      const noise = Math.sin(i / 3) * 50;
      const value = base + drift + noise;
      const cash = 1000 + Math.cos(i / 5) * 100;
      const investment = value - cash;
      return { date: t.toISOString().slice(0, 10), value: Math.round(value), cash: Math.round(cash), investment: Math.round(investment) };
    });
    return HttpResponse.json(points);
  }),
  
  // Health check endpoint (JSON for HealthPill)
  http.get('/api/health', () => 
    HttpResponse.json({ env: 'dev', gitSha: 'deadbeef', region: 'local', services: { api: { status: 'up', lastUpdated: new Date().toISOString() } }, asOf: new Date().toISOString() })
  ),
  http.get('/api/ping', () => HttpResponse.json({ status: 'ok' })),
  
    // Market data endpoints
  http.get('/api/quotes', ({ request }) => {
    const url = new URL(request.url);
    let symbols = (url.searchParams.get("symbols") || "").split(",").filter(Boolean);
    if (!symbols.length) {
      symbols = ['AAPL','MSFT','GOOGL','META','AMZN','NFLX'];
    }
    const quotes: Record<string, any> = {};

    const ensureList = symbols.length ? symbols : ["SPY","QQQ","AAPL","MSFT"]; 

    ensureList.forEach(symbol => {
      const seed = symbol.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
      const base = (seed % 1000) + 10; // $10–$1010
      const change = +(((seed % 7) - 3) * 0.12).toFixed(2);
      const last = +(base + change).toFixed(2);
      const changePct = +(((change / (last - change)) * 100)).toFixed(2);
      const bid = +(last - 0.05).toFixed(2);
      const ask = +(last + 0.05).toFixed(2);
      const prevClose = +(last - change).toFixed(2);
      const volume = 100000 + (seed % 900000);
      const t = new Date().toISOString();

      quotes[symbol] = {
        symbol,
        last,
        price: last,
        change,
        changePct,
        bid,
        ask,
        open: +(last - 0.8).toFixed(2),
        high: +(last + 1.2).toFixed(2),
        low:  +(last - 1.3).toFixed(2),
        prevClose,
        volume,
        time: t,
        quote: {
          bp: bid,
          ap: ask,
          bs: 100 + (seed % 900),
          as: 100 + (seed % 900),
          t
        },
        stale: false
      };
    });

    return HttpResponse.json({ quotes });
  }),
  
  http.get('/api/bars', ({ request }) => {
    const url = new URL(request.url);
    const symbol = url.searchParams.get("symbol") || "UNKNOWN";
    const timeframe = url.searchParams.get("timeframe") || "1Day";
    const limit = parseInt(url.searchParams.get("limit") || "100", 10);
    
    const seed = symbol.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const basePrice = (seed % 1000) + 10;
    const volatility = (seed % 10) / 100;
    
    const bars: Array<any> = [];
    const now = new Date();
    
    for (let i = 0; i < limit; i++) {
      const barTime = new Date(now);
      if (timeframe === "1Min") barTime.setMinutes(now.getMinutes() - i);
      else if (timeframe === "5Min") barTime.setMinutes(now.getMinutes() - i * 5);
      else if (timeframe === "15Min") barTime.setMinutes(now.getMinutes() - i * 15);
      else if (timeframe === "30Min") barTime.setMinutes(now.getMinutes() - i * 30);
      else if (timeframe === "1Hour") barTime.setHours(now.getHours() - i);
      else if (timeframe === "1Day") barTime.setDate(now.getDate() - i);
      else if (timeframe === "1Week") barTime.setDate(now.getDate() - i * 7);
      else if (timeframe === "1Month") barTime.setMonth(now.getMonth() - i);
      
      const dayFactor = Math.sin(i / 10) * volatility;
      const close = basePrice * (1 + dayFactor);
      const open = close * (1 + (Math.sin(i) * volatility / 2));
      const high = Math.max(open, close) * (1 + (volatility / 4));
      const low = Math.min(open, close) * (1 - (volatility / 4));
      const volume = Math.floor(100000 + (Math.sin(i) * 50000));
      
      bars.push({
        t: barTime.toISOString(),
        o: parseFloat(open.toFixed(2)),
        h: parseFloat(high.toFixed(2)),
        l: parseFloat(low.toFixed(2)),
        c: parseFloat(close.toFixed(2)),
        v: volume,
        n: Math.floor(volume / 100)
      });
    }
    
    return HttpResponse.json({
      bars: bars.reverse(),
      symbol,
      stale: false
    });
  }),
  
  // Health check endpoint
  http.get('/api/health/market', () => {
    return HttpResponse.json({
      status: "ok",
      code: 200,
      latency_ms: 42
    });
  }),
  
  // News sentiment endpoint
  http.get('/api/news/sentiment', ({ request }) => {
    const url = new URL(request.url);
    const category = url.searchParams.get('category') || 'markets';
    const query = url.searchParams.get('query') || '';
    
    const outlets: Record<string, { count: number; avg_sent: number; avg_partisan: number; avg_info: number }> = {
      'Reuters': { count: 3, avg_sent: 0.1, avg_partisan: 0.05, avg_info: 0.72 },
      'Bloomberg': { count: 2, avg_sent: 0.2, avg_partisan: 0.15, avg_info: 0.85 },
      'CNBC': { count: 4, avg_sent: -0.1, avg_partisan: 0.25, avg_info: 0.65 }
    };
    
    let clusters: Array<any> = [];
    
    if (category === 'markets') {
      clusters = [
        {
          headline: "Fed signals potential rate cuts in coming months",
          url: "https://example.com/fed-signals",
          sentiment: 0.45,
          partisan_spread: 0.2,
          informational: 0.85,
          finance: 0.9,
          sources: ["Reuters", "Bloomberg", "CNBC"],
          articles: []
        },
        {
          headline: "Tech stocks rally on earnings beats",
          url: "https://example.com/tech-rally",
          sentiment: 0.78,
          partisan_spread: 0.15,
          informational: 0.72,
          finance: 0.85,
          sources: ["Bloomberg", "CNBC"],
          articles: []
        },
        {
          headline: "Oil prices drop on supply concerns",
          url: "https://example.com/oil-prices",
          sentiment: -0.35,
          partisan_spread: 0.18,
          informational: 0.68,
          finance: 0.82,
          sources: ["Reuters"],
          articles: []
        }
      ];
    } else if (category === 'politics') {
      clusters = [
        {
          headline: "Senate passes new infrastructure bill",
          url: "https://example.com/infrastructure",
          sentiment: 0.25,
          partisan_spread: 0.65,
          informational: 0.55,
          finance: 0.45,
          sources: ["The Guardian (US)", "Fox News"],
          articles: []
        },
        {
          headline: "Trade negotiations stall with key partners",
          url: "https://example.com/trade-talks",
          sentiment: -0.3,
          partisan_spread: 0.7,
          informational: 0.6,
          finance: 0.5,
          sources: ["MSNBC", "Fox News"],
          articles: []
        }
      ];
    } else if (category === 'tech') {
      clusters = [
        {
          headline: "AI breakthrough announced by leading research lab",
          url: "https://example.com/ai-breakthrough",
          sentiment: 0.65,
          partisan_spread: 0.1,
          informational: 0.9,
          finance: 0.75,
          sources: ["TechCrunch"],
          articles: []
        }
      ];
    } else if (category === 'crypto') {
      clusters = [
        {
          headline: "Bitcoin surges past $50,000 on ETF approval news",
          url: "https://example.com/bitcoin-etf",
          sentiment: 0.8,
          partisan_spread: 0.25,
          informational: 0.7,
          finance: 0.95,
          sources: ["CoinDesk"],
          articles: []
        }
      ];
    } else if (category === 'macro') {
      clusters = [
        {
          headline: "GDP growth exceeds expectations in Q2",
          url: "https://example.com/gdp-growth",
          sentiment: 0.55,
          partisan_spread: 0.4,
          informational: 0.8,
          finance: 0.85,
          sources: ["The Economist", "Bloomberg"],
          articles: []
        },
        {
          headline: "Inflation data shows cooling trend",
          url: "https://example.com/inflation-cooling",
          sentiment: 0.6,
          partisan_spread: 0.35,
          informational: 0.75,
          finance: 0.8,
          sources: ["Reuters", "Bloomberg"],
          articles: []
        }
      ];
    }
    
    if (query) {
      clusters = clusters.filter(c => 
        c.headline.toLowerCase().includes(query.toLowerCase())
      );
    }
    
    return HttpResponse.json({
      category,
      clusters,
      outlets
    });
  }),
];
