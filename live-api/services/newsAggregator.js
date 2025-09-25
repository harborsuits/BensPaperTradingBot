// Multi-source news aggregator
class NewsAggregator {
  constructor() {
    this.sources = {
      newsdata: {
        enabled: !!process.env.NEWSDATA_API_KEY,
        key: process.env.NEWSDATA_API_KEY,
        baseUrl: 'https://newsdata.io/api/1/news',
        rateLimit: 200 // requests per day
      },
      gnews: {
        enabled: !!process.env.GNEWS_API_KEY,
        key: process.env.GNEWS_API_KEY,
        baseUrl: 'https://gnews.io/api/v4/search',
        rateLimit: 100 // requests per day
      },
      mediastack: {
        enabled: !!process.env.MEDIASTACK_API_KEY,
        key: process.env.MEDIASTACK_API_KEY,
        baseUrl: 'http://api.mediastack.com/v1/news',
        rateLimit: 500 // requests per month
      },
      currents: {
        enabled: !!process.env.CURRENTS_API_KEY,
        key: process.env.CURRENTS_API_KEY,
        baseUrl: 'https://api.currentsapi.services/v1/search',
        rateLimit: 600 // requests per day
      },
      nytimes: {
        enabled: !!process.env.NYTIMES_API_KEY,
        key: process.env.NYTIMES_API_KEY,
        baseUrl: 'https://api.nytimes.com/svc/search/v2/articlesearch.json',
        rateLimit: 500 // requests per day
      }
    };
    
    // Track usage to respect rate limits
    this.usage = new Map();
    this.resetUsageDaily();
  }

  resetUsageDaily() {
    // Reset usage counters daily
    setInterval(() => {
      this.usage.clear();
    }, 24 * 60 * 60 * 1000);
  }

  async fetchNews(query, options = {}) {
    const results = [];
    const errors = [];
    
    // Rotate through sources to distribute load
    const enabledSources = Object.entries(this.sources)
      .filter(([name, config]) => config.enabled)
      .sort(() => Math.random() - 0.5); // Randomize order
    
    // Try up to 3 sources to avoid rate limits
    const promises = enabledSources
      .slice(0, 3) // Only use 3 sources per request
      .map(async ([name, config]) => {
        try {
          const news = await this.fetchFromSource(name, config, query, options);
          return { source: name, news, error: null };
        } catch (error) {
          console.error(`[NewsAggregator] ${name} error:`, error.message);
          return { source: name, news: [], error: error.message };
        }
      });
    
    const sourceResults = await Promise.all(promises);
    
    // Combine and deduplicate results
    const allNews = [];
    const seenUrls = new Set();
    
    for (const { source, news, error } of sourceResults) {
      if (error) {
        errors.push({ source, error });
      }
      
      for (const item of news) {
        if (!seenUrls.has(item.url)) {
          seenUrls.add(item.url);
          allNews.push({
            ...item,
            source: source,
            reliability: this.getSourceReliability(source, item)
          });
        }
      }
    }
    
    // Sort by relevance and timestamp
    allNews.sort((a, b) => {
      // Prioritize by reliability
      if (a.reliability !== b.reliability) {
        return b.reliability - a.reliability;
      }
      // Then by timestamp
      return new Date(b.published_at) - new Date(a.published_at);
    });
    
    return {
      articles: allNews.slice(0, options.limit || 50),
      sources_used: sourceResults.filter(r => !r.error).map(r => r.source),
      errors: errors,
      total_found: allNews.length
    };
  }

  async fetchFromSource(sourceName, config, query, options) {
    // Check rate limit
    const usage = this.usage.get(sourceName) || 0;
    if (usage >= config.rateLimit * 0.8) { // 80% threshold
      throw new Error(`Rate limit approaching for ${sourceName}`);
    }
    this.usage.set(sourceName, usage + 1);
    
    switch (sourceName) {
      case 'newsdata':
        return this.fetchNewsdata(config, query, options);
      case 'gnews':
        return this.fetchGNews(config, query, options);
      case 'mediastack':
        return this.fetchMediastack(config, query, options);
      case 'currents':
        return this.fetchCurrents(config, query, options);
      case 'nytimes':
        return this.fetchNYTimes(config, query, options);
      default:
        return [];
    }
  }

  async fetchNewsdata(config, query, options) {
    const params = new URLSearchParams({
      apikey: config.key,
      q: query,
      language: 'en',
      category: 'business'
    });
    
    const response = await fetch(`${config.baseUrl}?${params}`);
    const data = await response.json();
    
    if (data.status !== 'success') {
      throw new Error(data.results?.message || 'Newsdata API error');
    }
    
    return (data.results || []).map(item => ({
      id: item.article_id,
      title: item.title,
      description: item.description,
      url: item.link,
      published_at: item.pubDate,
      source_name: item.source_id,
      sentiment: this.analyzeSentiment(item.title + ' ' + (item.description || '')),
      symbols: this.extractSymbols(item.title + ' ' + (item.description || ''))
    }));
  }

  async fetchGNews(config, query, options) {
    const params = new URLSearchParams({
      apikey: config.key,
      q: query,
      lang: 'en',
      max: options.limit || 10,
      in: 'title,description'
    });
    
    const response = await fetch(`${config.baseUrl}?${params}`);
    const data = await response.json();
    
    return (data.articles || []).map(item => ({
      id: item.url,
      title: item.title,
      description: item.description,
      url: item.url,
      published_at: item.publishedAt,
      source_name: item.source.name,
      sentiment: this.analyzeSentiment(item.title + ' ' + item.description),
      symbols: this.extractSymbols(item.title + ' ' + item.description)
    }));
  }

  async fetchMediastack(config, query, options) {
    const params = new URLSearchParams({
      access_key: config.key,
      keywords: query,
      languages: 'en',
      categories: 'business,finance',
      limit: options.limit || 10
    });
    
    const response = await fetch(`${config.baseUrl}?${params}`);
    const data = await response.json();
    
    return (data.data || []).map(item => ({
      id: item.url,
      title: item.title,
      description: item.description,
      url: item.url,
      published_at: item.published_at,
      source_name: item.source,
      sentiment: this.analyzeSentiment(item.title + ' ' + (item.description || '')),
      symbols: this.extractSymbols(item.title + ' ' + (item.description || ''))
    }));
  }

  async fetchCurrents(config, query, options) {
    const params = new URLSearchParams({
      apiKey: config.key,
      keywords: query,
      language: 'en',
      category: 'business'
    });
    
    const response = await fetch(`${config.baseUrl}?${params}`);
    const data = await response.json();
    
    if (data.status !== 'ok') {
      throw new Error('Currents API error');
    }
    
    return (data.news || []).map(item => ({
      id: item.id,
      title: item.title,
      description: item.description,
      url: item.url,
      published_at: item.published,
      source_name: item.author,
      sentiment: this.analyzeSentiment(item.title + ' ' + (item.description || '')),
      symbols: this.extractSymbols(item.title + ' ' + (item.description || ''))
    }));
  }

  async fetchNYTimes(config, query, options) {
    const params = new URLSearchParams({
      'api-key': config.key,
      q: query,
      fq: 'section_name:("Business" OR "Technology" OR "Economy")',
      sort: 'newest'
    });
    
    const response = await fetch(`${config.baseUrl}?${params}`);
    const data = await response.json();
    
    return (data.response?.docs || []).map(item => ({
      id: item._id,
      title: item.headline?.main,
      description: item.abstract || item.lead_paragraph,
      url: item.web_url,
      published_at: item.pub_date,
      source_name: 'New York Times',
      sentiment: this.analyzeSentiment(item.headline?.main + ' ' + (item.abstract || '')),
      symbols: this.extractSymbols(item.headline?.main + ' ' + (item.abstract || ''))
    }));
  }

  analyzeSentiment(text) {
    if (!text) return 0;
    
    const positive = ['surge', 'rally', 'gain', 'rise', 'beat', 'exceed', 'profit', 'growth', 'bullish', 'upgrade'];
    const negative = ['fall', 'drop', 'loss', 'decline', 'miss', 'weak', 'bear', 'crash', 'plunge', 'downgrade'];
    
    const lowerText = text.toLowerCase();
    let score = 0;
    
    positive.forEach(word => {
      if (lowerText.includes(word)) score += 0.1;
    });
    
    negative.forEach(word => {
      if (lowerText.includes(word)) score -= 0.1;
    });
    
    return Math.max(-1, Math.min(1, score));
  }

  extractSymbols(text) {
    if (!text) return [];
    
    // Match stock symbols (1-5 uppercase letters)
    const matches = text.match(/\b[A-Z]{1,5}\b/g) || [];
    
    // Common stock symbols to look for
    const knownSymbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CRM', 'NFLX'];
    
    return [...new Set(matches.filter(m => knownSymbols.includes(m)))];
  }

  getSourceReliability(source, article) {
    // Base reliability scores
    const baseScores = {
      nytimes: 0.9,
      newsdata: 0.7,
      gnews: 0.7,
      mediastack: 0.6,
      currents: 0.6
    };
    
    let score = baseScores[source] || 0.5;
    
    // Adjust based on article quality
    if (article.description && article.description.length > 100) score += 0.05;
    if (article.symbols && article.symbols.length > 0) score += 0.05;
    
    return Math.min(1, score);
  }

  async getMarketNews(limit = 20) {
    // Get general market news
    const marketTerms = 'stock market OR S&P 500 OR nasdaq OR dow jones OR trading';
    return this.fetchNews(marketTerms, { limit });
  }

  async getSymbolNews(symbol, limit = 10) {
    // Get news for specific symbol
    return this.fetchNews(symbol, { limit });
  }

  async getCryptoNews(limit = 10) {
    // Get crypto-specific news
    const cryptoTerms = 'bitcoin OR ethereum OR cryptocurrency OR crypto';
    return this.fetchNews(cryptoTerms, { limit });
  }

  async getMacroNews(limit = 10) {
    // Get macro economic news
    const macroTerms = 'federal reserve OR inflation OR GDP OR unemployment OR tariff OR interest rates';
    return this.fetchNews(macroTerms, { limit });
  }
}

module.exports = NewsAggregator;
