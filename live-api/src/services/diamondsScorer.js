/**
 * Diamonds Scorer - Computes news impact scores for trading signals
 * Uses news sentiment + price reaction + volume analysis
 */

const { recorder } = require('./marketRecorder');

class DiamondsScorer {
  constructor() {
    this.impactWeights = {
      sentiment: 0.4,
      volume: 0.3,
      priceReaction: 0.2,
      recency: 0.1
    };

    this.sentimentThresholds = {
      veryPositive: 0.3,
      positive: 0.1,
      neutral: -0.1,
      negative: -0.3,
      veryNegative: -0.5
    };
  }

  /**
   * Calculate impact score for a news item
   */
  async calculateImpactScore(newsItem, symbol) {
    try {
      const sentimentScore = this.calculateSentimentScore(newsItem.sentiment);
      const volumeScore = await this.calculateVolumeScore(symbol, newsItem.ts_feed);
      const priceScore = await this.calculatePriceReactionScore(symbol, newsItem.ts_feed);
      const recencyScore = this.calculateRecencyScore(newsItem.ts_feed);

      // Weighted combination
      const impactScore = (
        sentimentScore * this.impactWeights.sentiment +
        volumeScore * this.impactWeights.volume +
        priceScore * this.impactWeights.priceReaction +
        recencyScore * this.impactWeights.recency
      );

      return {
        symbol,
        impactScore: Math.max(0, Math.min(1, impactScore)), // Clamp to [0,1]
        components: {
          sentiment: sentimentScore,
          volume: volumeScore,
          priceReaction: priceScore,
          recency: recencyScore
        },
        evidence: {
          newsId: newsItem.id,
          headline: newsItem.headline,
          source: newsItem.source,
          url: newsItem.url,
          timestamp: newsItem.ts_feed
        }
      };
    } catch (error) {
      console.warn(`Failed to calculate impact score for ${symbol}:`, error.message);
      return {
        symbol,
        impactScore: 0,
        components: { sentiment: 0, volume: 0, priceReaction: 0, recency: 0 },
        evidence: {
          newsId: newsItem.id,
          headline: newsItem.headline,
          error: error.message
        }
      };
    }
  }

  /**
   * Calculate sentiment score component
   */
  calculateSentimentScore(sentiment) {
    const s = sentiment || 0;

    if (s >= this.sentimentThresholds.veryPositive) return 1.0;
    if (s >= this.sentimentThresholds.positive) return 0.7;
    if (s >= this.sentimentThresholds.neutral) return 0.5;
    if (s >= this.sentimentThresholds.negative) return 0.3;
    if (s >= this.sentimentThresholds.veryNegative) return 0.1;

    return 0.0;
  }

  /**
   * Calculate volume score component
   */
  async calculateVolumeScore(symbol, newsTimestamp) {
    try {
      // Get volume data around news timestamp
      const newsTime = new Date(newsTimestamp);
      const lookbackHours = 24;

      // Query for volume data in the period around news
      const volumeData = await this.getVolumeData(symbol, newsTime, lookbackHours);

      if (!volumeData || volumeData.length === 0) return 0.5; // Neutral

      // Calculate average volume before news
      const beforeNews = volumeData.filter(d => new Date(d.timestamp) < newsTime);
      const afterNews = volumeData.filter(d => new Date(d.timestamp) >= newsTime);

      if (beforeNews.length === 0 || afterNews.length === 0) return 0.5;

      const avgVolumeBefore = beforeNews.reduce((sum, d) => sum + d.volume, 0) / beforeNews.length;
      const avgVolumeAfter = afterNews.reduce((sum, d) => sum + d.volume, 0) / afterNews.length;

      // Volume spike indicates impact
      const volumeRatio = avgVolumeAfter / avgVolumeBefore;

      if (volumeRatio >= 2.0) return 1.0; // Massive volume spike
      if (volumeRatio >= 1.5) return 0.8; // Significant spike
      if (volumeRatio >= 1.2) return 0.6; // Moderate spike
      if (volumeRatio >= 0.8) return 0.4; // Normal volume
      return 0.2; // Below normal volume

    } catch (error) {
      console.warn(`Failed to calculate volume score: ${error.message}`);
      return 0.5; // Neutral score on error
    }
  }

  /**
   * Calculate price reaction score component
   */
  async calculatePriceReactionScore(symbol, newsTimestamp) {
    try {
      const newsTime = new Date(newsTimestamp);
      const reactionWindow = 60 * 60 * 1000; // 1 hour window

      // Get price data around news
      const priceData = await this.getPriceData(symbol, newsTime, reactionWindow);

      if (!priceData || priceData.length < 2) return 0.5;

      // Calculate price movement in reaction window
      const preNewsPrice = priceData[0]?.price || 0;
      const postNewsPrices = priceData.slice(1);
      const avgPostNewsPrice = postNewsPrices.reduce((sum, d) => sum + d.price, 0) / postNewsPrices.length;

      if (preNewsPrice === 0) return 0.5;

      const priceChange = (avgPostNewsPrice - preNewsPrice) / preNewsPrice;

      // Strong price reaction indicates impact
      if (Math.abs(priceChange) >= 0.05) return 1.0; // 5%+ move
      if (Math.abs(priceChange) >= 0.03) return 0.8; // 3%+ move
      if (Math.abs(priceChange) >= 0.02) return 0.6; // 2%+ move
      if (Math.abs(priceChange) >= 0.01) return 0.4; // 1%+ move
      return 0.2; // Minimal movement

    } catch (error) {
      console.warn(`Failed to calculate price reaction score: ${error.message}`);
      return 0.5;
    }
  }

  /**
   * Calculate recency score component
   */
  calculateRecencyScore(timestamp) {
    const newsTime = new Date(timestamp);
    const now = new Date();
    const hoursOld = (now - newsTime) / (1000 * 60 * 60);

    if (hoursOld <= 1) return 1.0; // Very recent
    if (hoursOld <= 6) return 0.8; // Recent
    if (hoursOld <= 12) return 0.6; // Somewhat recent
    if (hoursOld <= 24) return 0.4; // Day old
    if (hoursOld <= 72) return 0.2; // Few days old
    return 0.1; // Old news
  }

  /**
   * Get volume data for analysis (mock implementation)
   */
  async getVolumeData(symbol, newsTime, hoursBack) {
    // In production, this would query the market recorder for volume data
    // For now, return mock data
    const dataPoints = [];
    const interval = hoursBack * 60 * 60 * 1000 / 24; // Hourly data points

    for (let i = 0; i < 24; i++) {
      const timestamp = new Date(newsTime.getTime() - (hoursBack * 60 * 60 * 1000) + (i * interval));
      dataPoints.push({
        timestamp: timestamp.toISOString(),
        volume: Math.random() * 1000000 + 500000 // Random volume
      });
    }

    return dataPoints;
  }

  /**
   * Get price data for analysis (mock implementation)
   */
  async getPriceData(symbol, newsTime, windowMs) {
    // In production, this would query the market recorder for price data
    // For now, return mock data
    const dataPoints = [];
    const interval = windowMs / 10; // 10 data points in window

    for (let i = 0; i < 10; i++) {
      const timestamp = new Date(newsTime.getTime() + (i * interval));
      dataPoints.push({
        timestamp: timestamp.toISOString(),
        price: 100 + Math.random() * 20 // Random price around 100
      });
    }

    return dataPoints;
  }

  /**
   * Get top diamonds (high-impact news items)
   */
  async getTopDiamonds(limit = 10, minScore = 0.6) {
    try {
      // In production, this would query the news_snapshot table
      // For now, return mock high-impact items
      const mockDiamonds = [
        {
          symbol: 'AAPL',
          impactScore: 0.85,
          components: { sentiment: 0.8, volume: 0.9, priceReaction: 0.8, recency: 0.9 },
          evidence: {
            newsId: 'apple_earnings',
            headline: 'Apple Beats Q4 Earnings Estimates',
            source: 'Financial News',
            url: 'https://news.com/apple-earnings',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
          }
        },
        {
          symbol: 'TSLA',
          impactScore: 0.78,
          components: { sentiment: 0.7, volume: 0.8, priceReaction: 0.9, recency: 0.7 },
          evidence: {
            newsId: 'tesla_delivery',
            headline: 'Tesla Exceeds Delivery Expectations',
            source: 'Auto News',
            url: 'https://news.com/tesla-delivery',
            timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()
          }
        },
        {
          symbol: 'NVDA',
          impactScore: 0.72,
          components: { sentiment: 0.6, volume: 0.8, priceReaction: 0.7, recency: 0.8 },
          evidence: {
            newsId: 'nvidia_ai',
            headline: 'NVIDIA AI Chip Demand Surges',
            source: 'Tech News',
            url: 'https://news.com/nvidia-ai',
            timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString()
          }
        }
      ];

      return mockDiamonds
        .filter(diamond => diamond.impactScore >= minScore)
        .sort((a, b) => b.impactScore - a.impactScore)
        .slice(0, limit);

    } catch (error) {
      console.error('Failed to get top diamonds:', error);
      return [];
    }
  }
}

module.exports = { DiamondsScorer };
