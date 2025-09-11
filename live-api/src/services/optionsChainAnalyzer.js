/**
 * Options Chain Analyzer for EvoTester (JavaScript version)
 * Analyzes options chains for quality, liquidity, and trading opportunities
 */

class OptionsChainAnalyzer {
  /**
   * Analyze options chain quality and liquidity
   */
  analyzeChain(chain) {
    const spreadScore = this.calculateSpreadScore(chain);
    const volumeScore = this.calculateVolumeScore(chain);
    const oiScore = this.calculateOpenInterestScore(chain);

    const overall = (spreadScore * 0.4 + volumeScore * 0.35 + oiScore * 0.25);

    return {
      spreadScore,
      volumeScore,
      oiScore,
      overall
    };
  }

  /**
   * Calculate spread quality score (0-1, higher is better)
   */
  calculateSpreadScore(chain) {
    const allOptions = [...(chain.calls || []), ...(chain.puts || [])];
    if (allOptions.length === 0) return 0;

    const spreads = allOptions.map(opt => (opt.ask - opt.bid) / opt.bid);
    const avgSpread = spreads.reduce((sum, spread) => sum + spread, 0) / spreads.length;

    // Convert to score (lower spread = higher score)
    return Math.max(0, Math.min(1, 1 - (avgSpread / 0.1))); // 10% spread = 0 score
  }

  /**
   * Calculate volume quality score (0-1, higher is better)
   */
  calculateVolumeScore(chain) {
    const allOptions = [...(chain.calls || []), ...(chain.puts || [])];
    if (allOptions.length === 0) return 0;

    const volumes = allOptions.map(opt => opt.volume || 0);
    const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;

    // Score based on volume thresholds
    if (avgVolume >= 1000) return 1.0;
    if (avgVolume >= 500) return 0.8;
    if (avgVolume >= 100) return 0.6;
    if (avgVolume >= 50) return 0.4;
    if (avgVolume >= 10) return 0.2;
    return 0.0;
  }

  /**
   * Calculate open interest quality score (0-1, higher is better)
   */
  calculateOpenInterestScore(chain) {
    const allOptions = [...(chain.calls || []), ...(chain.puts || [])];
    if (allOptions.length === 0) return 0;

    const ois = allOptions.map(opt => opt.openInterest || 0);
    const avgOI = ois.reduce((sum, oi) => sum + oi, 0) / ois.length;

    // Score based on OI thresholds
    if (avgOI >= 5000) return 1.0;
    if (avgOI >= 2000) return 0.8;
    if (avgOI >= 1000) return 0.7;
    if (avgOI >= 500) return 0.6;
    if (avgOI >= 200) return 0.4;
    if (avgOI >= 50) return 0.2;
    return 0.0;
  }

  /**
   * Validate if chain meets minimum quality standards
   */
  validateChain(chain) {
    const score = this.analyzeChain(chain);
    const reasons = [];

    if (score.spreadScore < 0.5) {
      reasons.push('Poor spread quality');
    }

    if (score.volumeScore < 0.3) {
      reasons.push('Insufficient volume');
    }

    if (score.oiScore < 0.3) {
      reasons.push('Insufficient open interest');
    }

    if (score.overall < 0.5) {
      reasons.push('Overall chain quality too low');
    }

    return {
      valid: reasons.length === 0,
      reasons,
      score
    };
  }
}

module.exports = { OptionsChainAnalyzer };
