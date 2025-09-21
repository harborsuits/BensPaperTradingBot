/**
 * Real News Provider - Live news feed from financial APIs
 * Uses Finnhub API for real-time financial news and sentiment
 */

const axios = require('axios');

class RealNewsProvider {
  constructor(config = {}) {
    this.apiKey = process.env.FINNHUB_API_KEY || process.env.ALPHA_VANTAGE_API_KEY;
    this.baseUrl = 'https://finnhub.io/api/v1';
    this.alphaVantageUrl = 'https://www.alphavantage.co/query';
    this.rateLimit = 60; // Finnhub: 60 requests per minute
    this.lastRequest = 0;
    this.requestQueue = [];
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes cache
  }

  async enforceRateLimit() {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequest;
    const minInterval = (60 / this.rateLimit) * 1000; // milliseconds between requests

    if (timeSinceLastRequest < minInterval) {
      const waitTime = minInterval - timeSinceLastRequest;
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    this.lastRequest = Date.now();
  }

  async fetchFinnhubNews(sinceTs) {
    await this.enforceRateLimit();

    const sinceDate = new Date(sinceTs);
    const from = sinceDate.toISOString().split('T')[0]; // YYYY-MM-DD format

    try {
      const response = await axios.get(`${this.baseUrl}/news`, {
        params: {
          category: 'general',
          token: this.apiKey,
          from: from
        },
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        return response.data
          .filter(item => new Date(item.datetime * 1000) >= sinceDate)
          .map(item => this.normalizeFinnhubItem(item));
      }

      return [];
    } catch (error) {
      console.error('[RealNewsProvider] Finnhub API error:', error.message);
      return [];
    }
  }

  async fetchAlphaVantageNews(sinceTs) {
    await this.enforceRateLimit();

    try {
      const response = await axios.get(this.alphaVantageUrl, {
        params: {
          function: 'NEWS_SENTIMENT',
          apikey: this.apiKey,
          topics: 'financial_markets'
        },
        timeout: 10000
      });

      if (response.data && response.data.feed) {
        const sinceDate = new Date(sinceTs);
        return response.data.feed
          .filter(item => new Date(item.time_published) >= sinceDate)
          .map(item => this.normalizeAlphaVantageItem(item));
      }

      return [];
    } catch (error) {
      console.error('[RealNewsProvider] Alpha Vantage API error:', error.message);
      return [];
    }
  }

  normalizeFinnhubItem(item) {
    return {
      id: item.id?.toString() || `finnhub_${Date.now()}`,
      title: item.headline || item.title || 'Untitled',
      description: item.summary || item.description || '',
      publishedAt: new Date(item.datetime * 1000).toISOString(),
      source: item.source || 'Finnhub',
      url: item.url || '',
      symbols: item.related || [],
      sentiment: item.sentiment || null
    };
  }

  normalizeAlphaVantageItem(item) {
    return {
      id: item.title?.replace(/\s+/g, '_').toLowerCase() || `av_${Date.now()}`,
      title: item.title || 'Untitled',
      description: item.summary || '',
      publishedAt: item.time_published || new Date().toISOString(),
      source: item.source || 'Alpha Vantage',
      url: item.url || '',
      symbols: item.ticker_sentiment?.map(ts => ts.ticker) || [],
      sentiment: item.overall_sentiment_score || null
    };
  }

  async fetchSince(sinceTs) {
    if (!this.apiKey) {
      console.warn('[RealNewsProvider] No API key configured for real news data');
      return this.getFallbackNews(sinceTs);
    }

    try {
      // Try Finnhub first
      const finnhubNews = await this.fetchFinnhubNews(sinceTs);
      if (finnhubNews.length > 0) {
        console.log(`[RealNewsProvider] Fetched ${finnhubNews.length} news items from Finnhub`);
        return finnhubNews;
      }

      // Fallback to Alpha Vantage
      const alphaNews = await this.fetchAlphaVantageNews(sinceTs);
      if (alphaNews.length > 0) {
        console.log(`[RealNewsProvider] Fetched ${alphaNews.length} news items from Alpha Vantage`);
        return alphaNews;
      }

      // If both APIs fail, return minimal fallback
      console.warn('[RealNewsProvider] All news APIs failed, using minimal fallback');
      return this.getFallbackNews(sinceTs);

    } catch (error) {
      console.error('[RealNewsProvider] Critical error in fetchSince:', error);
      return this.getFallbackNews(sinceTs);
    }
  }

  /**
   * Minimal fallback for when APIs are unavailable
   */
  getFallbackNews(sinceTs) {
    const sinceDate = new Date(sinceTs);
    const now = new Date();

    if (now - sinceDate < 60 * 60 * 1000) { // Less than 1 hour old
      return [{
        id: `fallback_${Date.now()}`,
        title: 'Market Update',
        description: 'Real-time market data integration in progress. News feed will be available once API keys are configured.',
        publishedAt: now.toISOString(),
        source: 'System',
        url: '',
        symbols: ['SPY'],
        sentiment: null
      }];
    }

    return [];
  }
}

module.exports = { RealNewsProvider, MockNewsProvider };
