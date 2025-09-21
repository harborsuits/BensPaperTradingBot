export const POOR_CAPITAL_MODE = {
  risk: { perTradeRiskPct: 0.01, maxPositionNotionalPct: 0.10 },
  execution: { minStopDistanceBps: 20, maxSlippageBps: 25 },
  advancedGuards: { advParticipationMax: 0.05 },
  fitnessEnhancements: {
    capitalEfficiencyFloor: 0.2,
    frictionCap: 0.25
  },
  riskTilt: 1.0,
  leveragedETF: false,
  overnight: false,
  ttlRenew: false
};


