/**
 * Event Classifier - Rule-based news event detection
 * Fast, deterministic classification with statistical confidence scores
 */

class EventClassifier {
  constructor() {
    // Rule-based event patterns with confidence scores
    this.EVENT_RULES = [
      // Earnings events (high confidence)
      {
        type: 'earnings.surprise.up',
        confidence: 0.9,
        patterns: [
          /\b(beats|beat|exceeds|surpasses)\b.*\b(EPS|earnings|revenue)\b/i,
          /\b(raises|hikes|increases|upgrades?)\b.*\b(guidance|forecast|outlook)\b/i,
          /\b(earnings|EPS)\b.*\b(beat|surprise)\b.*\b(up|higher|above)\b/i
        ],
        direction: 1,
        sector_focus: ['technology', 'financials', 'healthcare']
      },
      {
        type: 'earnings.surprise.down',
        confidence: 0.9,
        patterns: [
          /\b(misses|miss|below|falls short)\b.*\b(EPS|earnings|revenue)\b/i,
          /\b(cuts?|lowers?|reduces?)\b.*\b(guidance|forecast|outlook)\b/i,
          /\b(earnings|EPS)\b.*\b(miss|disappointment|down|lower)\b/i
        ],
        direction: -1,
        sector_focus: ['technology', 'financials', 'healthcare']
      },

      // Macro/Fed events
      {
        type: 'macro.fed.hawkish',
        confidence: 0.7,
        patterns: [
          /\b(higher for longer|rate hike|hawkish|tightening)\b/i,
          /\b(Fed|Federal Reserve)\b.*\b(rate|rates)\b.*\b(increase|hike|raise)\b/i,
          /\b(interest rate|rates)\b.*\b(rise|increase|hike)\b/i
        ],
        direction: -1,
        sector_focus: ['financials', 'real_estate']
      },
      {
        type: 'macro.fed.dovish',
        confidence: 0.7,
        patterns: [
          /\b(rate cut|dovish|easing|pause)\b/i,
          /\b(Fed|Federal Reserve)\b.*\b(rate|rates)\b.*\b(cut|lower|reduce)\b/i,
          /\b(interest rate|rates)\b.*\b(fall|decrease|cut)\b/i
        ],
        direction: 1,
        sector_focus: ['technology', 'consumer_discretionary']
      },

      // Corporate events
      {
        type: 'corporate.mna.deal',
        confidence: 0.8,
        patterns: [
          /\b(to acquire|merger|acquisition|buyout|all-cash deal)\b/i,
          /\b(merges with|acquires|takes over)\b/i,
          /\b(deal|transaction|agreement)\b.*\b(announced|signed)\b/i
        ],
        direction: 1,
        sector_focus: ['technology', 'healthcare', 'financials']
      },

      // Regulatory events
      {
        type: 'regulatory.tailwind',
        confidence: 0.6,
        patterns: [
          /\b(approves?|clears?|greenlights?)\b.*\b(deal|merger|acquisition)\b/i,
          /\b(regulatory|SEC|FTC)\b.*\b(approval|clearance)\b/i,
          /\b(antitrust|regulatory)\b.*\b(concerns|issues)\b.*\b(resolved|solved)\b/i
        ],
        direction: 1,
        sector_focus: ['technology', 'healthcare', 'financials']
      },
      {
        type: 'regulatory.headwind',
        confidence: 0.6,
        patterns: [
          /\b(blocks?|rejects?|denies?)\b.*\b(deal|merger|acquisition)\b/i,
          /\b(regulatory|SEC|FTC)\b.*\b(concerns|investigation|probe)\b/i,
          /\b(antitrust|regulatory)\b.*\b(challenge|objection|opposition)\b/i
        ],
        direction: -1,
        sector_focus: ['technology', 'healthcare', 'financials']
      },

      // Supply chain events
      {
        type: 'supply.chain.tailwind',
        confidence: 0.5,
        patterns: [
          /\b(supply chain|logistics|shipping)\b.*\b(improves?|resolves?|stabilizes?)\b/i,
          /\b(chip|semiconductor)\b.*\b(shortage|crisis)\b.*\b(eases?|ends?)\b/i,
          /\b(container|shipping)\b.*\b(rates|costs)\b.*\b(fall|decrease|drop)\b/i
        ],
        direction: 1,
        sector_focus: ['technology', 'industrials', 'consumer_discretionary']
      },
      {
        type: 'supply.chain.headwind',
        confidence: 0.5,
        patterns: [
          /\b(supply chain|logistics|shipping)\b.*\b(disrupts?|delays?|problems?)\b/i,
          /\b(chip|semiconductor)\b.*\b(shortage|crisis|bottleneck)\b/i,
          /\b(container|shipping)\b.*\b(rates|costs)\b.*\b(rise|increase|surge)\b/i
        ],
        direction: -1,
        sector_focus: ['technology', 'industrials', 'consumer_discretionary']
      },

      // Sector readthrough events
      {
        type: 'sector.readthrough.positive',
        confidence: 0.4,
        patterns: [
          /\b(supplier|vendor|partner)\b.*\b(benefits?|gains?|wins?)\b/i,
          /\b(customer|client)\b.*\b(orders?|contracts?)\b.*\b(increase|rises?)\b/i,
          /\b(industry|sector)\b.*\b(ripple effect|contagion|spread)\b.*\b(positive|beneficial)\b/i
        ],
        direction: 1,
        sector_focus: ['all']
      },
      {
        type: 'sector.readthrough.negative',
        confidence: 0.4,
        patterns: [
          /\b(supplier|vendor|partner)\b.*\b(impacts?|hurts?|loses?)\b/i,
          /\b(customer|client)\b.*\b(orders?|contracts?)\b.*\b(cancel|reduce|delay)\b/i,
          /\b(industry|sector)\b.*\b(ripple effect|contagion|spread)\b.*\b(negative|harmful)\b/i
        ],
        direction: -1,
        sector_focus: ['all']
      }
    ];

    // Performance tracking
    this.performanceStats = {
      headlinesProcessed: 0,
      eventsDetected: 0,
      avgLatency: 0,
      errorRate: 0
    };
  }

  /**
   * Classify a news headline into event types
   * @param {string} headline - News headline to classify
   * @param {string} source - News source (for quality weighting)
   * @param {Array<string>} tickers - Associated ticker symbols
   * @returns {Array<EventSignal>} Detected events with confidence scores
   */
  classify(headline, source = 'unknown', tickers = []) {
    const startTime = Date.now();

    try {
      const events = [];
      const lowerHeadline = headline.toLowerCase();

      // Check each rule
      for (const rule of this.EVENT_RULES) {
        let maxMatchStrength = 0;
        let bestMatch = null;

        // Test all patterns for this rule
        for (const pattern of rule.patterns) {
          const matches = lowerHeadline.match(pattern);
          if (matches) {
            // Calculate match strength based on pattern specificity
            const matchStrength = this._calculateMatchStrength(matches, headline);
            if (matchStrength > maxMatchStrength) {
              maxMatchStrength = matchStrength;
              bestMatch = matches;
            }
          }
        }

        if (maxMatchStrength > 0) {
          // Apply context guards and quality adjustments
          const adjustedConfidence = this._adjustConfidence(
            rule.confidence * maxMatchStrength,
            source,
            tickers,
            headline
          );

          if (adjustedConfidence >= 0.3) { // Minimum threshold
            events.push({
              type: rule.type,
              direction: rule.direction,
              confidence: adjustedConfidence,
              tickers: tickers,
              sectorFocus: rule.sector_focus,
              headline: headline,
              source: source,
              timestamp: new Date().toISOString(),
              matchStrength: maxMatchStrength,
              validated: false, // Will be set by validation layer
              effectZ: 0 // Will be populated by reaction stats
            });
          }
        }
      }

      // Update performance stats
      const latency = Date.now() - startTime;
      this._updatePerformanceStats(events.length > 0, latency);

      return events;

    } catch (error) {
      console.error('EventClassifier error:', error);
      this.performanceStats.errorRate = (this.performanceStats.errorRate + 1) / (this.performanceStats.headlinesProcessed + 1);
      return [];
    }
  }

  /**
   * Calculate match strength based on pattern specificity and context
   */
  _calculateMatchStrength(matches, headline) {
    let strength = 0.5; // Base strength

    // Increase for longer matches (more specific)
    if (matches[0]) {
      strength += Math.min(matches[0].length / headline.length, 0.3);
    }

    // Increase for multiple keywords
    const keywordCount = matches.length - 1; // First match is full string
    strength += Math.min(keywordCount * 0.1, 0.2);

    // Boost for key financial terms
    const financialTerms = /\b(EPS|earnings|revenue|guidance|forecast|beat|miss|hike|cut|raise|lower)\b/gi;
    const financialMatches = headline.match(financialTerms);
    if (financialMatches) {
      strength += Math.min(financialMatches.length * 0.05, 0.15);
    }

    return Math.min(strength, 1.0);
  }

  /**
   * Adjust confidence based on source quality, ticker context, and other factors
   */
  _adjustConfidence(baseConfidence, source, tickers, headline) {
    let adjusted = baseConfidence;

    // Source quality adjustment
    const sourceQuality = this._getSourceQuality(source);
    adjusted *= sourceQuality;

    // Ticker presence boost
    if (tickers && tickers.length > 0) {
      adjusted *= 1.1; // Small boost for having tickers
    }

    // Length and clarity boost
    if (headline.length > 50 && headline.length < 200) {
      adjusted *= 1.05;
    }

    // Time sensitivity boost (breaking news indicators)
    const urgentTerms = /\b(breaking|urgent|flash|alert|just announced|just reported)\b/i;
    if (urgentTerms.test(headline)) {
      adjusted *= 1.1;
    }

    return Math.min(adjusted, 1.0);
  }

  /**
   * Get source quality multiplier
   */
  _getSourceQuality(source) {
    const qualityMap = {
      'bloomberg': 1.1,
      'reuters': 1.1,
      'wsj': 1.1,
      'cnbc': 1.0,
      'yahoo': 0.9,
      'marketwatch': 0.9,
      'seekingalpha': 0.8,
      'benzinga': 0.8,
      'finviz': 0.7,
      'unknown': 0.6
    };

    const lowerSource = source.toLowerCase();
    return qualityMap[lowerSource] || qualityMap['unknown'];
  }

  /**
   * Update performance statistics
   */
  _updatePerformanceStats(detectedEvents, latency) {
    this.performanceStats.headlinesProcessed++;

    if (detectedEvents) {
      this.performanceStats.eventsDetected++;
    }

    // Rolling average latency
    const alpha = 0.1; // Smoothing factor
    this.performanceStats.avgLatency =
      (1 - alpha) * this.performanceStats.avgLatency + alpha * latency;
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats() {
    return {
      ...this.performanceStats,
      detectionRate: this.performanceStats.headlinesProcessed > 0 ?
        this.performanceStats.eventsDetected / this.performanceStats.headlinesProcessed : 0
    };
  }

  /**
   * Reset performance statistics
   */
  resetPerformanceStats() {
    this.performanceStats = {
      headlinesProcessed: 0,
      eventsDetected: 0,
      avgLatency: 0,
      errorRate: 0
    };
  }
}

module.exports = { EventClassifier };
