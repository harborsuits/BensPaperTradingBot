// live-api/src/services/BrainService.js
const axios = require('axios');

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
      return {
        symbol: decision.symbol,
        experts: [
          { name: "technical", score: decision.confidence || 0.5, conf: 0.8 },
          { name: "news", score: avgSentiment, conf: 0.7 },
          { name: "volatility", score: decision.confidence || 0.5, conf: 0.6 },
          { name: "statistical", score: decision.confidence || 0.5, conf: 0.75 }
        ],
        final_score: decision.confidence || 0.5,
        conf: decision.confidence || 0.5,
        world_model: {
          regime: 'neutral',
          volatility: 'medium',
          trend: 'up',
          market_sentiment: avgSentiment
        },
        policy_used_id: "python_brain",
        snapshot_ts: snapshot_ts || "NOW",
        processing_ms: data.processing_time_ms || 50,
        news_sentiment: avgSentiment,
        outlets_used: Object.keys(outlets).length,
        fallback: false
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

module.exports = {
  scoreSymbol,
  planTrade
};
