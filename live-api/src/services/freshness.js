// live-api/src/services/freshness.js

function currentFreshness() {
  return {
    quotes_s: 1.5, // Mock realistic fresh data
    features_s: 4.2,
    world_model_s: 8.1,
  };
}

module.exports = { currentFreshness };
