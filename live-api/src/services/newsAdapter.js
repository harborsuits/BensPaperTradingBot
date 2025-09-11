/**
 * Enhanced NewsAdapter - Modern AI-powered news sentiment analysis
 * Supports multiple news sources with advanced NLP and ML capabilities
 */

const natural = require('natural');
const aposToLexForm = require('apos-to-lex-form');
const SpellCorrector = require('spelling-corrector');
const SW = require('stopword');

class EnhancedNewsAdapter {
  constructor(config = {}) {
    this.config = {
      apiKey: config.apiKey || process.env.NEWS_API_KEY,
      baseUrl: config.baseUrl,
      rateLimit: config.rateLimit || 100, // requests per minute
      cacheExpiry: config.cacheExpiry || 300000, // 5 minutes
      useAI: config.useAI !== false, // Enable AI analysis by default
      modelPath: config.modelPath || './models/sentiment-model',
      ...config
    };

    this.lastRequest = 0;
    this.requestCount = 0;
    this.cache = new Map();

    // Initialize NLP components
    this.sentimentAnalyzer = new natural.SentimentAnalyzer('English', natural.PorterStemmer, 'afinn');
    this.spellCorrector = new SpellCorrector();
    this.spellCorrector.loadDictionary();

    // Enhanced sentiment lexicons
    this.positiveWords = [
      'upgrade', 'beat', 'surge', 'rally', 'gain', 'rise', 'bullish', 'buy', 'strong', 'growth',
      'outperform', 'soar', 'jump', 'climb', 'breakout', 'momentum', 'bull', 'long', 'call',
      'buyback', 'acquisition', 'merger', 'expansion', 'earnings beat', 'revenue growth',
      'profit surge', 'market share gain', 'technological breakthrough', 'innovation'
    ];

    this.negativeWords = [
      'downgrade', 'miss', 'plunge', 'fall', 'drop', 'bearish', 'sell', 'weak', 'decline', 'crash',
      'underperform', 'tumble', 'slump', 'collapse', 'bankruptcy', 'lawsuit', 'scandal', 'recall',
      'bear', 'short', 'put', 'profit warning', 'revenue miss', 'market share loss', 'layoffs'
    ];

    // Context multipliers for more accurate sentiment
    this.contextMultipliers = {
      'earnings': { positive: 1.5, negative: 1.8 },
      'revenue': { positive: 1.4, negative: 1.6 },
      'guidance': { positive: 1.3, negative: 1.7 },
      'forecast': { positive: 1.2, negative: 1.5 },
      'outlook': { positive: 1.3, negative: 1.4 },
      'margin': { positive: 1.2, negative: 1.3 }
    };

    // Market sector analysis
    this.sectorKeywords = {
      'technology': ['tech', 'software', 'semiconductor', 'ai', 'cloud', 'cybersecurity'],
      'healthcare': ['pharma', 'biotech', 'medical', 'fda', 'clinical trial'],
      'finance': ['bank', 'financial', 'fed', 'interest rate', 'mortgage'],
      'energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'coal'],
      'consumer': ['retail', 'ecommerce', 'consumer goods', 'automotive']
    };

    this.sentimentHistory = new Map(); // Track sentiment trends
  }

  /**
   * Enhanced sentiment analysis with context awareness
   */
  analyzeSentiment(item) {
    const text = `${item.title || ''} ${item.description || ''}`.toLowerCase();

    // Clean and normalize text
    const lexedReview = aposToLexForm(text);
    const casedReview = lexedReview.toLowerCase();
    const alphaOnlyReview = casedReview.replace(/[^a-zA-Z\s]+/g, '');

    // Remove stopwords
    const { WordTokenizer } = natural;
    const tokenizer = new WordTokenizer();
    const tokenizedReview = tokenizer.tokenize(alphaOnlyReview);
    const filteredReview = SW.removeStopwords(tokenizedReview);

    // Spell correction
    const spellCorrectedReview = filteredReview.map(word => this.spellCorrector.correct(word));

    // Calculate base sentiment score
    let baseScore = 0;
    let positiveHits = 0;
    let negativeHits = 0;

    spellCorrectedReview.forEach(word => {
      if (this.positiveWords.some(pos => word.includes(pos))) {
        positiveHits++;
        baseScore += 0.15;
      }
      if (this.negativeWords.some(neg => word.includes(neg))) {
        negativeHits++;
        baseScore -= 0.15;
      }
    });

    // Apply context multipliers
    let contextMultiplier = 1.0;
    Object.entries(this.contextMultipliers).forEach(([context, multipliers]) => {
      if (text.includes(context)) {
        if (baseScore > 0) {
          contextMultiplier *= multipliers.positive;
        } else if (baseScore < 0) {
          contextMultiplier *= multipliers.negative;
        }
      }
    });

    // Apply intensity based on word proximity and repetition
    const intensityMultiplier = Math.min(1.5, 1 + (positiveHits + negativeHits) * 0.1);

    // Use natural library for additional analysis
    let nlpScore = 0;
    if (this.config.useAI && filteredReview.length > 0) {
      try {
        nlpScore = this.sentimentAnalyzer.getSentiment(filteredReview);
        nlpScore = nlpScore * 0.3; // Weight NLP score lower
      } catch (error) {
        console.warn('NLP sentiment analysis failed:', error.message);
      }
    }

    // Combine scores
    let finalScore = (baseScore * contextMultiplier * intensityMultiplier) + nlpScore;

    // Detect market sector
    const sector = this.detectSector(text);

    // Calculate confidence based on evidence strength
    const confidence = Math.min(0.95, Math.max(0.1,
      (positiveHits + negativeHits) / Math.max(10, spellCorrectedReview.length)
    ));

    // Clamp to [-1, 1]
    finalScore = Math.max(-1, Math.min(1, finalScore));

    // Store in sentiment history for trend analysis
    this.updateSentimentHistory(item.symbols || [], finalScore, confidence);

    return {
      score: finalScore,
      confidence: confidence,
      magnitude: Math.abs(finalScore),
      sector: sector,
      evidence: {
        positiveHits,
        negativeHits,
        totalWords: spellCorrectedReview.length,
        contextMultiplier: contextMultiplier.toFixed(2),
        intensityMultiplier: intensityMultiplier.toFixed(2)
      }
    };
  }

  /**
   * Detect market sector from text
   */
  detectSector(text) {
    const textLower = text.toLowerCase();
    for (const [sector, keywords] of Object.entries(this.sectorKeywords)) {
      if (keywords.some(keyword => textLower.includes(keyword))) {
        return sector;
      }
    }
    return 'general';
  }

  /**
   * Enhanced symbol extraction with better filtering
   */
  extractSymbols(item) {
    const text = `${item.title || ''} ${item.description || ''}`.toUpperCase();

    // Enhanced regex patterns for different symbol formats
    const patterns = [
      /\b([A-Z]{1,5})\b/g,                    // Standard symbols
      /\b([A-Z]{2,5})\s*\.\s*([A-Z]{1,2})\b/g, // With exchange suffix
      /\b\$([A-Z]{1,5})\b/g                    // With dollar sign
    ];

    const allMatches = [];
    patterns.forEach(pattern => {
      const matches = text.match(pattern) || [];
      allMatches.push(...matches);
    });

    // Enhanced exclusions with more financial terms
    const exclusions = [
      'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY',
      'HOT', 'BUT', 'SHE', 'HAS', 'CEO', 'CFO', 'CTO', 'COO', 'IPO', 'EPS', 'GDP', 'GDP', 'FED', 'NYSE', 'NASDAQ'
    ];

    // Filter and validate symbols
    const symbols = allMatches
      .map(match => match.replace(/[\$\.]/g, '').trim()) // Clean formatting
      .filter(symbol => {
        // Must be 2-5 characters, start with letter, no exclusions
        return symbol.length >= 2 &&
               symbol.length <= 5 &&
               /^[A-Z]/.test(symbol) &&
               !exclusions.includes(symbol);
      })
      .filter((symbol, index, arr) => arr.indexOf(symbol) === index); // dedupe

    // Additional validation for known symbols (could integrate with symbol database)
    return symbols.filter(symbol => this.isValidSymbol(symbol));
  }

  /**
   * Validate symbol format and basic rules
   */
  isValidSymbol(symbol) {
    // Basic validation - could be enhanced with actual symbol database lookup
    const validPatterns = [
      /^[A-Z]{2,5}$/,           // Standard format
      /^[A-Z]+\.[A-Z]+$/,       // With exchange
      /^\$[A-Z]{2,5}$/          // With dollar sign
    ];

    return validPatterns.some(pattern => pattern.test(symbol)) &&
           !/^[AEIOU]{3,}/.test(symbol) && // Avoid too many vowels (likely not a symbol)
           !/(.)\1{2,}/.test(symbol);      // Avoid repeated characters
  }

  /**
   * Track sentiment trends over time
   */
  updateSentimentHistory(symbols, score, confidence) {
    const now = Date.now();

    symbols.forEach(symbol => {
      if (!this.sentimentHistory.has(symbol)) {
        this.sentimentHistory.set(symbol, []);
      }

      const history = this.sentimentHistory.get(symbol);
      history.push({
        timestamp: now,
        score: score,
        confidence: confidence
      });

      // Keep only last 100 entries per symbol
      if (history.length > 100) {
        history.shift();
      }
    });
  }

  /**
   * Get sentiment trend for a symbol
   */
  getSentimentTrend(symbol, hours = 24) {
    if (!this.sentimentHistory.has(symbol)) {
      return null;
    }

    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    const history = this.sentimentHistory.get(symbol)
      .filter(entry => entry.timestamp >= cutoff);

    if (history.length < 2) {
      return null;
    }

    const recent = history.slice(-10); // Last 10 entries
    const avgScore = recent.reduce((sum, entry) => sum + entry.score, 0) / recent.length;
    const avgConfidence = recent.reduce((sum, entry) => sum + entry.confidence, 0) / recent.length;

    const trend = recent.length > 1 ?
      (recent[recent.length - 1].score - recent[0].score) / recent.length : 0;

    return {
      symbol,
      averageScore: avgScore,
      averageConfidence: avgConfidence,
      trend,
      sampleSize: recent.length,
      timeRange: `${hours}h`
    };
  }

  /**
   * Enhanced news item normalization with AI insights
   */
  normalizeItem(item) {
    const sentiment = this.analyzeSentiment(item);
    const symbols = this.extractSymbols(item);
    const trend = symbols.length > 0 ? this.getSentimentTrend(symbols[0]) : null;

    return {
      id: item.id || item.url || `${item.source}_${Date.now()}`,
      ts_feed: item.publishedAt || item.timestamp || new Date().toISOString(),
      source: item.source || this.constructor.name.toLowerCase(),
      headline: item.title || item.headline,
      summary: item.description || item.summary || '',
      url: item.url || item.link,
      symbols: symbols,
      sentiment: sentiment.score,
      sentimentDetails: sentiment,
      sector: sentiment.sector,
      trend: trend,
      relevance: this.calculateRelevance(item, symbols),
      raw: item
    };
  }

  /**
   * Calculate news relevance score
   */
  calculateRelevance(item, symbols) {
    let relevance = 0.5; // Base relevance

    // Higher relevance for items with symbols
    if (symbols.length > 0) relevance += 0.2;

    // Higher relevance for financial keywords
    const financialTerms = ['earnings', 'revenue', 'profit', 'guidance', 'forecast', 'acquisition', 'merger'];
    const text = `${item.title || ''} ${item.description || ''}`.toLowerCase();
    const financialMatches = financialTerms.filter(term => text.includes(term)).length;
    relevance += financialMatches * 0.1;

    // Higher relevance for major sources
    const majorSources = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'ft'];
    const source = (item.source || '').toLowerCase();
    if (majorSources.some(major => source.includes(major))) {
      relevance += 0.15;
    }

    return Math.min(1.0, relevance);
  }
}

// Maintain backward compatibility
class NewsAdapter extends EnhancedNewsAdapter {
  constructor(config = {}) {
    super(config);
    // Disable AI features for backward compatibility unless explicitly enabled
    this.config.useAI = config.useAI || false;
  }
}

module.exports = { NewsAdapter };
