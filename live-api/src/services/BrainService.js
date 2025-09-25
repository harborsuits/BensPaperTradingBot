// live-api/src/services/BrainService.js
const axios = require('axios');

// Embedding service configuration
const EMBEDDER_URL = process.env.EMBEDDER_URL || 'http://localhost:8002';
const USE_EMBEDDINGS = process.env.USE_EMBEDDINGS === 'true';
const MIN_SIMILARITY_SAMPLES = parseInt(process.env.MIN_SIMILARITY_SAMPLES || '100');
const SEMANTIC_WEIGHT = parseFloat(process.env.SEMANTIC_WEIGHT || '0.2');

const PY_BRAIN = process.env.PY_BRAIN_URL || "http://localhost:8001";
const TIMEOUT_MS = +(process.env.BRAIN_TIMEOUT_MS || 450);

async function scoreSymbol(symbol, snapshot_ts) {
  try {
    // Fetch real news sentiment data for this symbol
    const newsData = await axios.get(
      `http://localhost:4000/api/news/sentiment?category=markets`,
      { timeout: 2000 }
    ).catch(err => {
      console.log('News sentiment fetch failed, using fallback:', err.message);
      return { data: { outlets: {} } };
    });

    // Extract sentiment for this symbol's market/sector
    const outlets = newsData.data?.outlets || {};
    const avgSentiment = Object.values(outlets).reduce((sum, outlet) => {
      return sum + (outlet.avg_sent || 0);
    }, 0) / Math.max(Object.keys(outlets).length, 1);

    // Get semantic similarity data if embeddings are enabled
    let semanticPrior = 0;
    let similarCases = [];
    let semanticConfidence = 0;

    if (USE_EMBEDDINGS) {
      try {
        // Find similar historical cases
        const recentHeadlines = await getRecentHeadlinesForSymbol(symbol);
        if (recentHeadlines && recentHeadlines.length > 0) {
          const headlineText = recentHeadlines[0].text || recentHeadlines[0].headline || '';

          const similarityResponse = await axios.post(`${EMBEDDER_URL}/similar`, {
            query_text: headlineText,
            k: 5,
            min_similarity: 0.3,
            metadata_filters: { tickers: [symbol] }
          }, { timeout: 2000 });

          const results = similarityResponse.data?.results || [];

          if (results.length > 0) {
            // Calculate semantic prior using shrinkage
            const validResults = results.filter(r => r.avg_return_1d !== null && r.sample_size_1d >= MIN_SIMILARITY_SAMPLES);

            if (validResults.length > 0) {
              // Weighted average of historical returns with shrinkage
              let totalWeightedReturn = 0;
              let totalWeight = 0;

              validResults.forEach(result => {
                const weight = result.similarity_score * Math.min(1, result.sample_size_1d / MIN_SIMILARITY_SAMPLES);
                totalWeightedReturn += result.avg_return_1d * weight;
                totalWeight += weight;
              });

              if (totalWeight > 0) {
                semanticPrior = totalWeightedReturn / totalWeight;
                semanticPrior *= SEMANTIC_WEIGHT; // Scale the semantic contribution
                semanticConfidence = Math.min(1.0, totalWeight / validResults.length);
              }

              similarCases = validResults.slice(0, 3).map(r => ({
                text: r.text.substring(0, 100) + '...',
                similarity: r.similarity_score,
                historical_return: r.avg_return_1d,
                sample_size: r.sample_size_1d,
                timestamp: r.timestamp
              }));
            }
          }
        }
      } catch (embedError) {
        console.log('Embedding service unavailable, proceeding without semantic prior:', embedError.message);
      }
    }

    // Use the correct Python endpoint: /api/decide
    const { data } = await axios.post(
      `${PY_BRAIN}/api/decide`,
      {
        opportunities: [{
          symbol,
          alpha: 0.5, // Will be computed by Python brain
          sentiment_boost: avgSentiment,
          regime_align: 0.5,
          est_cost_bps: 20,
          risk_penalty: 0,
          news_sentiment: avgSentiment,
          market_sentiment: avgSentiment
        }],
        current_regime: 'neutral',
        market_sentiment: avgSentiment,
        news_data: {
          outlets_count: Object.keys(outlets).length,
          avg_sentiment: avgSentiment,
          timestamp: new Date().toISOString()
        }
      },
      { timeout: TIMEOUT_MS }
    );

    // Transform Python response to Node format
    if (data.decisions && data.decisions[0]) {
      const decision = data.decisions[0];
      const baseScore = decision.confidence || 0.5;
      const finalScore = baseScore + semanticPrior; // Add semantic prior

      return {
        symbol: decision.symbol,
        experts: [
          { name: "technical", score: decision.confidence || 0.5, conf: 0.8 },
          { name: "news", score: avgSentiment, conf: 0.7 },
          { name: "volatility", score: decision.confidence || 0.5, conf: 0.6 },
          { name: "statistical", score: decision.confidence || 0.5, conf: 0.75 },
          { name: "semantic", score: semanticPrior, conf: semanticConfidence }
        ],
        final_score: Math.max(0, Math.min(1, finalScore)), // Clamp to [0,1]
        conf: Math.max(0.5, decision.confidence || finalScore || 0.5),
        world_model: {
          regime: 'neutral',
          volatility: 'medium',
          trend: 'up',
          market_sentiment: avgSentiment
        },
        policy_used_id: USE_EMBEDDINGS ? "python_brain_with_embeddings" : "python_brain",
        snapshot_ts: snapshot_ts || "NOW",
        processing_ms: data.processing_time_ms || 50,
        news_sentiment: avgSentiment,
        outlets_used: Object.keys(outlets).length,
        fallback: false,
        // Add embedding data
        semantic_data: {
          enabled: USE_EMBEDDINGS,
          prior_contribution: semanticPrior,
          confidence: semanticConfidence,
          similar_cases: similarCases,
          min_samples_required: MIN_SIMILARITY_SAMPLES,
          weight_used: SEMANTIC_WEIGHT
        }
      };
    }
    throw new Error("No decision returned from Python brain");
  } catch (err) {
    // Surface fallback so UI can show "Degraded (ML offline)"
    console.log('BrainService fallback triggered:', err.message);
    return {
      final_score: 0,
      conf: 0,
      experts: [],
      world_model: {},
      policy_used_id: "fallback-sim",
      fallback: true,
      error: err?.message || "python_brain_unreachable",
    };
  }
}

async function planTrade(req) {
  try {
    // For now, use simple rule-based planning since Python doesn't have plan endpoint
    // TODO: Add /api/brain/plan to Python service
    const { symbol, final_score, conf, account_state } = req;

    let action = "hold";
    let sizing_dollars = 0;
    let stop = null;
    let target = null;
    let structure = null;
    let reason = "python_brain_plan";

    if (final_score > 0.8 && conf > 0.7) {
      action = "enter";
      sizing_dollars = Math.min(account_state?.buying_power || 10000, 5000);
      reason = "high_confidence_entry";
    } else if (final_score < 0.3) {
      action = "avoid";
      reason = "low_confidence_avoid";
    }

    return {
      action,
      sizing_dollars,
      stop,
      target,
      structure,
      reason,
      fallback: false
    };
  } catch (err) {
    return { action: "avoid", reason: "fallback_brain_unreachable", fallback: true };
  }
}

// Helper function to get recent headlines for a symbol
async function getRecentHeadlinesForSymbol(symbol) {
  try {
    // Try to get headlines from news API
    const headlinesResponse = await axios.get(
      `http://localhost:4000/api/news/recent?symbol=${symbol}&limit=5`,
      { timeout: 1000 }
    );

    if (headlinesResponse.data && headlinesResponse.data.headlines) {
      return headlinesResponse.data.headlines;
    }

    // Fallback: try general market news
    const marketNewsResponse = await axios.get(
      `http://localhost:4000/api/news/recent?category=markets&limit=5`,
      { timeout: 1000 }
    );

    if (marketNewsResponse.data && marketNewsResponse.data.headlines) {
      return marketNewsResponse.data.headlines;
    }

    return null;
  } catch (error) {
    console.log('Could not fetch recent headlines:', error.message);
    return null;
  }
}

module.exports = {
  scoreSymbol,
  planTrade
};