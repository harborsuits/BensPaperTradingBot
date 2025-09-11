// live-api/src/services/policy.js

async function withinEarningsWindow(symbol) {
  // TODO: Implement actual earnings calendar lookup
  // For now, return false (no earnings window active)
  return false;
}

module.exports = { withinEarningsWindow };
