/**
 * Macro Event Analyzer
 * Detects and analyzes macro events (tariffs, regulations, political signals)
 * and their cascading effects on sectors and positions
 */

class MacroEventAnalyzer {
  constructor() {
    // Macro event patterns with sector impacts
    this.macroPatterns = {
      'tariff': {
        keywords: ['tariff', 'trade war', 'import duties', 'trade sanctions', 'customs'],
        certaintyMarkers: {
          high: ['will impose', 'announcing', 'effective immediately'],
          medium: ['considering', 'may impose', 'reviewing'],
          low: ['could', 'might consider', 'discussing']
        },
        sectorImpacts: {
          'steel': { direct: -5, timeline: 'immediate' },
          'aluminum': { direct: -5, timeline: 'immediate' },
          'automotive': { direct: -3, timeline: '1-3 days' },
          'technology': { direct: -4, timeline: '1-7 days' },
          'agriculture': { direct: -6, timeline: '3-7 days' },
          'retail': { direct: -2, timeline: '7-14 days' },
          'industrial': { direct: -3, timeline: '1-3 days' }
        },
        beneficiaries: {
          'domestic_steel': +3,
          'defense': +2,
          'infrastructure': +1
        }
      },
      
      'fed_policy': {
        keywords: ['fed', 'federal reserve', 'interest rate', 'fomc', 'monetary policy'],
        certaintyMarkers: {
          high: ['decided to', 'will raise', 'will cut'],
          medium: ['likely to', 'expected to', 'signals'],
          low: ['may consider', 'discussing', 'monitoring']
        },
        sectorImpacts: {
          'banking': { direct: +2, timeline: 'immediate' },
          'real_estate': { direct: -3, timeline: '1-7 days' },
          'utilities': { direct: -2, timeline: '1-3 days' },
          'technology': { direct: -4, timeline: 'immediate' },
          'consumer_discretionary': { direct: -2, timeline: '3-7 days' }
        }
      },
      
      'regulation': {
        keywords: ['regulation', 'antitrust', 'investigation', 'probe', 'compliance'],
        certaintyMarkers: {
          high: ['launching investigation', 'filing suit', 'new rules'],
          medium: ['considering action', 'reviewing', 'concerns about'],
          low: ['may look into', 'could investigate', 'monitoring']
        },
        sectorImpacts: {
          'technology': { direct: -5, timeline: 'immediate' },
          'healthcare': { direct: -3, timeline: '1-7 days' },
          'financial': { direct: -4, timeline: '1-3 days' },
          'energy': { direct: -3, timeline: '3-14 days' }
        }
      },
      
      'political_controversy': {
        keywords: ['boycott', 'controversy', 'backlash', 'protest', 'scandal'],
        certaintyMarkers: {
          high: ['widespread calls', 'major boycott', 'trending #1'],
          medium: ['growing backlash', 'calls for', 'criticism'],
          low: ['some criticism', 'concerns raised', 'questions about']
        },
        sectorImpacts: {
          'consumer_brands': { direct: -5, timeline: 'immediate' },
          'retail': { direct: -3, timeline: '1-3 days' },
          'entertainment': { direct: -4, timeline: 'immediate' },
          'social_media': { direct: -2, timeline: '1-7 days' }
        }
      }
    };
    
    // Track speaker reliability
    this.speakerReliability = new Map();
    
    // Historical pattern outcomes
    this.historicalPatterns = new Map();
  }
  
  /**
   * Analyze news event for macro implications
   */
  analyzeMacroEvent(newsItem, currentPositions = []) {
    const pattern = this.identifyPattern(newsItem);
    if (!pattern) return null;
    
    const certainty = this.assessCertainty(newsItem, pattern);
    const speaker = this.extractSpeaker(newsItem);
    const speakerWeight = this.getSpeakerReliability(speaker);
    
    // Calculate weighted probability
    const probability = certainty.level * speakerWeight;
    
    // Identify affected sectors in portfolio
    const exposures = this.calculateSectorExposure(currentPositions);
    const impactedPositions = this.getImpactedPositions(exposures, pattern);
    
    return {
      event_type: pattern.type,
      pattern_name: pattern.name,
      probability: probability,
      certainty: certainty,
      speaker: {
        name: speaker,
        reliability: speakerWeight,
        track_record: this.speakerReliability.get(speaker)
      },
      affected_sectors: pattern.sectorImpacts,
      portfolio_impact: {
        immediate_risk: this.calculateImmediateRisk(impactedPositions, pattern),
        positions_at_risk: impactedPositions,
        suggested_actions: this.getSuggestedActions(pattern, probability, impactedPositions)
      },
      timeline: this.getEventTimeline(pattern),
      historical_similar: this.findHistoricalMatches(pattern.type)
    };
  }
  
  /**
   * Identify which macro pattern this news matches
   */
  identifyPattern(newsItem) {
    const text = (newsItem.headline + ' ' + newsItem.description).toLowerCase();
    
    for (const [patternName, pattern] of Object.entries(this.macroPatterns)) {
      const keywordMatches = pattern.keywords.filter(keyword => 
        text.includes(keyword)
      ).length;
      
      if (keywordMatches > 0) {
        return {
          name: patternName,
          type: patternName,
          ...pattern,
          matchStrength: keywordMatches / pattern.keywords.length
        };
      }
    }
    
    return null;
  }
  
  /**
   * Assess certainty level of the event
   */
  assessCertainty(newsItem, pattern) {
    const text = (newsItem.headline + ' ' + newsItem.description).toLowerCase();
    
    for (const [level, markers] of Object.entries(pattern.certaintyMarkers)) {
      for (const marker of markers) {
        if (text.includes(marker)) {
          return {
            level: level === 'high' ? 0.9 : level === 'medium' ? 0.6 : 0.3,
            marker: marker,
            category: level
          };
        }
      }
    }
    
    return { level: 0.5, marker: 'none', category: 'unknown' };
  }
  
  /**
   * Extract speaker from news item
   */
  extractSpeaker(newsItem) {
    // Simple extraction - in production would use NLP
    const patterns = [
      /said\s+([A-Z][a-z]+\s+[A-Z][a-z]+)/,
      /([A-Z][a-z]+\s+[A-Z][a-z]+)\s+said/,
      /according to\s+([A-Z][a-z]+\s+[A-Z][a-z]+)/
    ];
    
    for (const pattern of patterns) {
      const match = newsItem.description.match(pattern);
      if (match) return match[1];
    }
    
    return newsItem.source || 'Unknown';
  }
  
  /**
   * Get speaker reliability score
   */
  getSpeakerReliability(speaker) {
    const record = this.speakerReliability.get(speaker);
    if (!record) return 0.5; // Unknown speaker = 50% weight
    
    return record.correct / record.total;
  }
  
  /**
   * Update speaker reliability after outcome
   */
  updateSpeakerReliability(speaker, prediction, outcome) {
    if (!this.speakerReliability.has(speaker)) {
      this.speakerReliability.set(speaker, { correct: 0, total: 0 });
    }
    
    const record = this.speakerReliability.get(speaker);
    record.total++;
    
    if (this.predictionMatched(prediction, outcome)) {
      record.correct++;
    }
    
    console.log(`[MacroAnalyzer] ${speaker} reliability: ${((record.correct/record.total)*100).toFixed(1)}%`);
  }
  
  /**
   * Calculate sector exposure in portfolio
   */
  calculateSectorExposure(positions) {
    const exposures = new Map();
    
    for (const position of positions) {
      const sector = this.getSymbolSector(position.symbol);
      const currentExposure = exposures.get(sector) || { value: 0, symbols: [] };
      
      currentExposure.value += position.value || (position.qty * position.price);
      currentExposure.symbols.push(position.symbol);
      
      exposures.set(sector, currentExposure);
    }
    
    return exposures;
  }
  
  /**
   * Get positions impacted by macro event
   */
  getImpactedPositions(exposures, pattern) {
    const impacted = [];
    
    for (const [sector, exposure] of exposures) {
      const impact = pattern.sectorImpacts[sector];
      if (impact) {
        impacted.push({
          sector,
          symbols: exposure.symbols,
          exposure_value: exposure.value,
          expected_impact: impact.direct,
          timeline: impact.timeline
        });
      }
    }
    
    return impacted.sort((a, b) => Math.abs(b.expected_impact) - Math.abs(a.expected_impact));
  }
  
  /**
   * Calculate immediate portfolio risk
   */
  calculateImmediateRisk(impactedPositions, pattern) {
    let totalRisk = 0;
    let immediateRisk = 0;
    
    for (const position of impactedPositions) {
      const risk = position.exposure_value * (position.expected_impact / 100);
      totalRisk += risk;
      
      if (position.timeline === 'immediate') {
        immediateRisk += risk;
      }
    }
    
    return {
      immediate_dollar_risk: immediateRisk,
      total_dollar_risk: totalRisk,
      risk_percentage: (totalRisk / this.getTotalPortfolioValue()) * 100
    };
  }
  
  /**
   * Get suggested actions based on macro event
   */
  getSuggestedActions(pattern, probability, impactedPositions) {
    const actions = [];
    
    // High probability = more aggressive action
    if (probability > 0.7) {
      actions.push({
        action: 'EXIT_POSITIONS',
        urgency: 'HIGH',
        positions: impactedPositions.filter(p => p.expected_impact < -3),
        reason: 'High probability negative event'
      });
    } else if (probability > 0.5) {
      actions.push({
        action: 'REDUCE_EXPOSURE',
        urgency: 'MEDIUM',
        positions: impactedPositions.filter(p => p.expected_impact < -2),
        target_reduction: 0.5,
        reason: 'Moderate probability negative event'
      });
    }
    
    // Look for opportunities
    if (pattern.beneficiaries) {
      actions.push({
        action: 'CONSIDER_LONGS',
        urgency: 'LOW',
        sectors: Object.keys(pattern.beneficiaries),
        reason: 'Potential beneficiaries of macro event'
      });
    }
    
    // Suggest hedges
    if (impactedPositions.length > 0) {
      actions.push({
        action: 'HEDGE',
        urgency: probability > 0.6 ? 'HIGH' : 'MEDIUM',
        suggestion: this.getHedgeSuggestion(pattern, impactedPositions),
        reason: 'Portfolio protection'
      });
    }
    
    return actions;
  }
  
  /**
   * Get hedge suggestions
   */
  getHedgeSuggestion(pattern, impactedPositions) {
    // Simple hedge logic - in production would be more sophisticated
    if (pattern.type === 'tariff') {
      return {
        instrument: 'XLI_PUTS', // Industrial sector puts
        rationale: 'Industrial sector typically falls on tariff news'
      };
    } else if (pattern.type === 'fed_policy') {
      return {
        instrument: 'TLT_CALLS', // Bond ETF calls
        rationale: 'Bonds typically rise when stocks fall on rate news'
      };
    }
    
    return {
      instrument: 'VXX_CALLS',
      rationale: 'Volatility hedge for uncertain macro event'
    };
  }
  
  /**
   * Simple sector mapping (would be more comprehensive in production)
   */
  getSymbolSector(symbol) {
    const sectorMap = {
      // Tech
      'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
      'META': 'technology', 'NVDA': 'technology', 'AMD': 'technology',
      
      // Financial
      'JPM': 'banking', 'BAC': 'banking', 'WFC': 'banking',
      'GS': 'banking', 'MS': 'banking',
      
      // Industrial
      'BA': 'industrial', 'CAT': 'industrial', 'GE': 'industrial',
      'MMM': 'industrial', 'HON': 'industrial',
      
      // Consumer
      'AMZN': 'retail', 'WMT': 'retail', 'TGT': 'retail',
      'NKE': 'consumer_brands', 'MCD': 'consumer_brands',
      
      // Materials
      'X': 'steel', 'NUE': 'steel', 'AA': 'aluminum',
      
      // Real Estate
      'SPG': 'real_estate', 'O': 'real_estate', 'VNQ': 'real_estate',
      
      // Default
      'SPY': 'broad_market', 'QQQ': 'technology'
    };
    
    return sectorMap[symbol] || 'unknown';
  }
  
  // Placeholder - would connect to actual portfolio value
  getTotalPortfolioValue() {
    return 100000; // Would fetch from broker
  }
  
  predictionMatched(prediction, outcome) {
    // Simple matching logic
    return Math.abs(prediction - outcome) < 0.2;
  }
  
  getEventTimeline(pattern) {
    return {
      immediate: '0-24 hours',
      short_term: '1-7 days',
      medium_term: '1-4 weeks',
      impacts: pattern.sectorImpacts
    };
  }
  
  findHistoricalMatches(eventType) {
    // Would query historical database
    return this.historicalPatterns.get(eventType) || [];
  }
}

module.exports = MacroEventAnalyzer;
