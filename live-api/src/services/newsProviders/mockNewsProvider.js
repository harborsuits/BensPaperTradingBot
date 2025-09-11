/**
 * Mock News Provider - Simulated news feed for testing
 * In production, replace with real news APIs (Finnhub, Alpha Vantage, etc.)
 */

const { NewsAdapter } = require('../newsAdapter');

class MockNewsProvider extends NewsAdapter {
  constructor(config = {}) {
    super({
      baseUrl: 'https://mock-news-api.example.com',
      rateLimit: 60, // 60 requests per minute
      ...config
    });

    // Mock news database
    this.mockNews = [
      {
        id: 'mock_001',
        title: 'Apple Reports Strong Q4 Earnings, Beats Estimates',
        description: 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue growth.',
        publishedAt: new Date(Date.now() - 2 * 60 * 1000).toISOString(), // 2 minutes ago
        source: 'Mock Financial News',
        url: 'https://mocknews.com/apple-earnings-q4'
      },
      {
        id: 'mock_002',
        title: 'Tesla Stock Surges on EV Demand Data',
        description: 'Tesla shares rose sharply after reporting higher-than-expected electric vehicle demand in key markets.',
        publishedAt: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5 minutes ago
        source: 'Mock Market Watch',
        url: 'https://mocknews.com/tesla-ev-demand'
      },
      {
        id: 'mock_003',
        title: 'Microsoft Announces Major AI Partnership',
        description: 'Microsoft Corp. announced a strategic partnership with leading AI research institutions to advance machine learning capabilities.',
        publishedAt: new Date(Date.now() - 8 * 60 * 1000).toISOString(), // 8 minutes ago
        source: 'Mock Tech News',
        url: 'https://mocknews.com/microsoft-ai-partnership'
      },
      {
        id: 'mock_004',
        title: 'SPY ETF Sees Record Inflows Amid Market Volatility',
        description: 'The SPDR S&P 500 ETF Trust saw record investor inflows as traders sought safe-haven assets during recent market turbulence.',
        publishedAt: new Date(Date.now() - 10 * 60 * 1000).toISOString(), // 10 minutes ago
        source: 'Mock ETF Report',
        url: 'https://mocknews.com/spy-etf-inflows'
      },
      {
        id: 'mock_005',
        title: 'NVIDIA Faces Supply Chain Challenges',
        description: 'NVIDIA Corporation reported supply chain disruptions that may impact graphics card availability in the coming quarter.',
        publishedAt: new Date(Date.now() - 12 * 60 * 1000).toISOString(), // 12 minutes ago
        source: 'Mock Supply Chain News',
        url: 'https://mocknews.com/nvidia-supply-chain'
      }
    ];
  }

  async fetchSince(sinceTs) {
    // Enforce rate limiting
    await this.enforceRateLimit();

    const sinceDate = new Date(sinceTs);

    // Filter mock news by timestamp
    const recentNews = this.mockNews.filter(item => {
      const itemDate = new Date(item.publishedAt);
      return itemDate >= sinceDate;
    });

    // Add some randomization to simulate real API behavior
    const randomizedNews = recentNews.map(item => ({
      ...item,
      publishedAt: new Date(item.publishedAt).toISOString(),
      // Add some fake recent items
      ...(Math.random() > 0.7 ? {
        title: item.title + ' (Updated)',
        publishedAt: new Date(Date.now() - Math.random() * 60 * 60 * 1000).toISOString()
      } : {})
    }));

    // Normalize all items
    return randomizedNews.map(item => this.normalizeItem(item));
  }

  /**
   * Generate additional mock news for testing
   */
  generateAdditionalNews(count = 5) {
    const companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ'];
    const actions = ['Surges', 'Declines', 'Reports', 'Announces', 'Faces', 'Sees'];
    const contexts = ['Earnings', 'Partnership', 'Challenge', 'Growth', 'Demand', 'Supply'];

    for (let i = 0; i < count; i++) {
      const company = companies[Math.floor(Math.random() * companies.length)];
      const action = actions[Math.floor(Math.random() * actions.length)];
      const context = contexts[Math.floor(Math.random() * contexts.length)];

      this.mockNews.push({
        id: `mock_${Date.now()}_${i}`,
        title: `${company} ${action} on ${context} News`,
        description: `${company} ${action.toLowerCase()} following ${context.toLowerCase()} developments in the market.`,
        publishedAt: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
        source: 'Mock Real-time News',
        url: `https://mocknews.com/${company.toLowerCase()}-${action.toLowerCase()}-${context.toLowerCase()}`
      });
    }
  }
}

module.exports = { MockNewsProvider };
