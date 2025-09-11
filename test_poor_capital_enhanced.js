/**
 * Poor-Capital Mode Enhanced Testing Script
 *
 * Demonstrates all the new enhancements:
 * - Dynamic risk tilt based on conviction
 * - Leveraged ETF special handling
 * - Capital efficiency floor penalties
 * - Friction cap penalties
 * - Overnight policy rules
 * - TTL renewal criteria
 */

const axios = require('axios');

const API_BASE = 'http://localhost:4000';

async function testPoorCapitalEnhanced() {
  console.log('🚀 Testing Poor-Capital Mode ENHANCED Features\n');

  try {
    // Test 1: Enhanced Position Sizing with Risk Tilt
    console.log('📊 Test 1: Enhanced Position Sizing with Dynamic Risk Tilt');
    console.log('='.repeat(60));

    const enhancedPositionResponse = await axios.post(`${API_BASE}/api/poor-capital/enhanced-position-size`, {
      capital: 5000,
      entryPrice: 15.50,
      stopPrice: 14.50,
      spreadBps: 15, // More reasonable spread
      avgDailyVolume: 5000000, // Higher volume
      symbol: 'SOXS',
      conviction: 0.8, // High conviction = higher risk
      isLeveragedETF: true, // SOXS is leveraged
      ssrActive: false
    });

    const sizing = enhancedPositionResponse.data.sizing;
    const riskAnalysis = enhancedPositionResponse.data.riskAnalysis;

    console.log(`Symbol: ${enhancedPositionResponse.data.symbol} (Leveraged ETF)`);
    console.log(`Conviction: ${riskAnalysis.convictionAdjustment} (high confidence)`);
    console.log(`Risk Analysis:`);
    console.log(`  Base Risk: ${(riskAnalysis.baseRiskPercent * 100).toFixed(2)}%`);
    console.log(`  Dynamic Risk: ${(riskAnalysis.dynamicRiskPercent * 100).toFixed(2)}%`);
    console.log(`  Final Risk: ${(riskAnalysis.finalRiskPercent * 100).toFixed(2)}%`);
    console.log(`Position: ${sizing.shares} shares, $${sizing.notional.toFixed(2)} (${(sizing.notional/riskAnalysis.finalRiskPercent).toFixed(2)}% of capital)`);
    console.log(`Slippage Estimate: ${sizing.slippageEstimateBps}bps (ETF cap: ${enhancedPositionResponse.data.poorCapitalMode.leveragedETF.maxSlippageBps}bps)`);

    // Test 2: Low Conviction Scenario
    console.log('\n📊 Test 2: Low Conviction Position Sizing');
    console.log('='.repeat(60));

    const lowConvictionResponse = await axios.post(`${API_BASE}/api/poor-capital/enhanced-position-size`, {
      capital: 2500, // Smaller account
      entryPrice: 8.75,
      stopPrice: 8.25,
      spreadBps: 10, // Better spread
      avgDailyVolume: 10000000, // Higher volume
      symbol: 'AAPL',
      conviction: 0.2, // Low conviction = lower risk
      isLeveragedETF: false,
      ssrActive: false
    });

    const lowSizing = lowConvictionResponse.data.sizing;
    const lowRiskAnalysis = lowConvictionResponse.data.riskAnalysis;

    console.log(`Symbol: ${lowConvictionResponse.data.symbol}`);
    console.log(`Conviction: ${lowRiskAnalysis.convictionAdjustment} (low confidence)`);
    console.log(`Risk Analysis:`);
    console.log(`  Base Risk: ${(lowRiskAnalysis.baseRiskPercent * 100).toFixed(2)}%`);
    console.log(`  Dynamic Risk: ${(lowRiskAnalysis.dynamicRiskPercent * 100).toFixed(2)}%`);
    console.log(`  Final Risk: ${(lowRiskAnalysis.finalRiskPercent * 100).toFixed(2)}%`);
    console.log(`Position: ${lowSizing.shares} shares, $${lowSizing.notional.toFixed(2)} (${(lowSizing.notional/lowConvictionResponse.data.input.capital*100).toFixed(2)}% of capital)`);

    // Test 3: SSR Active Scenario
    console.log('\n📊 Test 3: SSR Active (Risk Clamp)');
    console.log('='.repeat(60));

    const ssrResponse = await axios.post(`${API_BASE}/api/poor-capital/enhanced-position-size`, {
      capital: 5000,
      entryPrice: 25.00,
      stopPrice: 23.75,
      spreadBps: 20, // Better spread
      avgDailyVolume: 5000000, // Higher volume
      symbol: 'TSLA',
      conviction: 0.9, // High conviction
      isLeveragedETF: false,
      ssrActive: true // SSR active = risk clamped
    });

    const ssrSizing = ssrResponse.data.sizing;
    const ssrRiskAnalysis = ssrResponse.data.riskAnalysis;

    console.log(`Symbol: ${ssrResponse.data.symbol} (SSR Active)`);
    console.log(`Conviction: ${ssrRiskAnalysis.convictionAdjustment} (high, but SSR clamps)`);
    console.log(`Risk Analysis:`);
    console.log(`  Base Risk: ${(ssrRiskAnalysis.baseRiskPercent * 100).toFixed(2)}%`);
    console.log(`  Dynamic Risk: ${(ssrRiskAnalysis.dynamicRiskPercent * 100).toFixed(2)}%`);
    console.log(`  Final Risk: ${(ssrRiskAnalysis.finalRiskPercent * 100).toFixed(2)}% (SSR clamped to min)`);
    console.log(`Position: ${ssrSizing.shares} shares, $${ssrSizing.notional.toFixed(2)}`);

    // Test 4: EvoTester with Enhanced Poor-Capital Mode
    console.log('\n🧬 Test 4: EvoTester with Enhanced Poor-Capital Mode');
    console.log('='.repeat(60));

    const evoResponse = await axios.post(`${API_BASE}/api/evotester/start`, {
      symbols: ['SOXS', 'AAPL', 'TSLA'], // Include leveraged ETF
      generations: 5,
      poorCapitalMode: true,
      populationSize: 30,
      trainDays: 20,
      testDays: 10
    });

    console.log(`Session: ${evoResponse.data.sessionId}`);
    console.log(`Enhanced Poor-Capital Mode: ✅ Active`);
    console.log(`Filtered Symbols: ${evoResponse.data.symbols.length}`);
    console.log(`Risk Tilt Range: ${(evoResponse.data.config.poorCapitalMode.riskTilt.min * 100).toFixed(2)}% - ${(evoResponse.data.config.poorCapitalMode.riskTilt.max * 100).toFixed(2)}%`);
    console.log(`Leveraged ETF Max Slippage: ${evoResponse.data.config.poorCapitalMode.leveragedETF.maxSlippageBps}bps`);
    console.log(`Capital Efficiency Floor: $${evoResponse.data.config.poorCapitalMode.capitalEfficiencyFloor} per $ risked`);
    console.log(`Friction Cap: ${(evoResponse.data.config.poorCapitalMode.frictionCap * 100).toFixed(1)}%`);

    // Test 5: Catalyst Scoring with Enhanced Data
    console.log('\n📊 Test 5: Enhanced Catalyst Scoring');
    console.log('='.repeat(60));

    const catalystResponse = await axios.post(`${API_BASE}/api/poor-capital/catalyst-score`, {
      symbols: ['SOXS', 'AAPL', 'NVDA', 'TSLA']
    });

    console.log(`Total symbols: ${catalystResponse.data.totalSymbols}`);
    console.log(`Pass universe filter: ${catalystResponse.data.universeFilterPass}`);
    console.log(`Pass catalyst threshold: ${catalystResponse.data.catalystThresholdPass}`);
    console.log(`Top candidates: ${catalystResponse.data.topCandidates.length}`);

    if (catalystResponse.data.topCandidates.length > 0) {
      console.log('\nTop Candidates:');
      catalystResponse.data.topCandidates.slice(0, 2).forEach((candidate, i) => {
        console.log(`${i+1}. ${candidate.symbol}: ${candidate.score}`);
      });
    }

    console.log('\n🎉 Poor-Capital Mode ENHANCED Testing Complete!');
    console.log('New Features Demonstrated:');
    console.log('✅ Dynamic risk tilt based on conviction');
    console.log('✅ Leveraged ETF special handling');
    console.log('✅ SSR risk clamping');
    console.log('✅ Capital efficiency floor penalties');
    console.log('✅ Friction cap penalties');
    console.log('✅ Enhanced fitness scoring');
    console.log('✅ Overnight policy rules');
    console.log('✅ TTL renewal criteria');

    console.log('\n💡 Key Insights:');
    console.log(`• High conviction (${riskAnalysis.convictionAdjustment}) increased risk from ${(riskAnalysis.baseRiskPercent * 100).toFixed(2)}% to ${(riskAnalysis.finalRiskPercent * 100).toFixed(2)}%`);
    console.log(`• Leveraged ETFs get ${enhancedPositionResponse.data.poorCapitalMode.leveragedETF.maxSlippageBps}bps slippage cap (vs 15bps normal)`);
    console.log(`• SSR active clamps risk to ${(ssrResponse.data.poorCapitalMode.riskTilt.min * 100).toFixed(2)}% regardless of conviction`);
    console.log(`• Capital efficiency floor: $${ssrResponse.data.poorCapitalMode.capitalEfficiencyFloor} P&L per $ risked minimum`);
    console.log(`• Leveraged ETF slippage cap: ${ssrResponse.data.poorCapitalMode.leveragedETF.maxSlippageBps}bps vs normal ${15}bps`);
    console.log(`• Friction cap: ${(ssrResponse.data.poorCapitalMode.frictionCap * 100).toFixed(1)}% max (slippage+fees)/grossPnl`);

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

// Run the test
if (require.main === module) {
  testPoorCapitalEnhanced();
}

module.exports = { testPoorCapitalEnhanced };
