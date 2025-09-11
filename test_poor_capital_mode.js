/**
 * Poor-Capital Mode Testing Script
 *
 * Demonstrates the three-ring approach:
 * 1. Universe filtering with catalyst scoring
 * 2. Position sizing with realistic constraints
 * 3. Fitness enhancement with capital efficiency
 */

const axios = require('axios');

const API_BASE = 'http://localhost:4000';

async function testPoorCapitalMode() {
  console.log('🤑 Testing Poor-Capital Mode Implementation\n');

  try {
    // Test 1: Catalyst Scoring
    console.log('📊 Test 1: Catalyst Scoring');
    console.log('=' .repeat(50));

    const symbols = ['AAPL', 'NVDA', 'TSLA', 'PLTR', 'SOXS', 'TQQQ', 'SPY', 'QQQ', 'IWM', 'ARKK'];

    const catalystResponse = await axios.post(`${API_BASE}/api/poor-capital/catalyst-score`, {
      symbols
    });

    console.log(`Input symbols: ${symbols.length}`);
    console.log(`Pass universe filter: ${catalystResponse.data.universeFilterPass}`);
    console.log(`Pass catalyst threshold: ${catalystResponse.data.catalystThresholdPass}`);
    console.log(`Top candidates: ${catalystResponse.data.topCandidates.length}\n`);

    console.log('Top 3 Candidates:');
    catalystResponse.data.topCandidates.slice(0, 3).forEach((candidate, i) => {
      console.log(`${i+1}. ${candidate.symbol}: ${candidate.score} (News: ${candidate.newsImpact}, RVOL: ${candidate.relativeVolume})`);
    });

    // Test 2: Position Sizing
    console.log('\n💰 Test 2: Position Sizing');
    console.log('=' .repeat(50));

    const topSymbol = catalystResponse.data.topCandidates[0]?.symbol || 'AAPL';

    const positionResponse = await axios.post(`${API_BASE}/api/poor-capital/position-size`, {
      capital: 5000, // $5K starting capital
      entryPrice: 15.50,
      stopPrice: 14.50, // 6.5% stop
      spreadBps: 25,
      avgDailyVolume: 500000,
      symbol: topSymbol
    });

    const sizing = positionResponse.data.sizing;
    console.log(`Symbol: ${positionResponse.data.symbol}`);
    console.log(`Capital: $${positionResponse.data.input.capital}`);
    console.log(`Entry: $${positionResponse.data.input.entryPrice}, Stop: $${positionResponse.data.input.stopPrice}`);
    console.log(`Shares: ${sizing.shares} (${sizing.canExecute ? '✅ Executable' : '❌ Rejected'})`);

    if (sizing.canExecute) {
      console.log(`Notional: $${sizing.notional.toFixed(2)} (${(sizing.notional/positionResponse.data.input.capital*100).toFixed(1)}% of capital)`);
      console.log(`Risk: ${(sizing.riskPercent*100).toFixed(2)}% ($${sizing.riskAmount.toFixed(2)})`);
      console.log(`ADV Participation: ${(sizing.advParticipationPercent*100).toFixed(3)}%`);
      console.log(`Slippage Estimate: ${sizing.slippageEstimateBps}bps`);
    } else {
      console.log(`Rejection: ${sizing.rejectionReason}`);
    }

    // Test 3: EvoTester with Poor-Capital Mode
    console.log('\n🧬 Test 3: EvoTester with Poor-Capital Mode');
    console.log('=' .repeat(50));

    const evoResponse = await axios.post(`${API_BASE}/api/evotester/start`, {
      symbols: ['all'], // Use all available symbols
      generations: 10,
      poorCapitalMode: true, // Enable Poor-Capital Mode
      populationSize: 50,
      trainDays: 30,
      testDays: 15
    });

    console.log(`Session: ${evoResponse.data.sessionId}`);
    console.log(`Poor-Capital Mode: ${evoResponse.data.poorCapitalMode ? '✅ Enabled' : '❌ Disabled'}`);
    console.log(`Filtered Symbols: ${evoResponse.data.symbols.length}`);
    console.log(`Risk per Trade: ${(evoResponse.data.config.poorCapitalMode?.riskPerTrade * 100).toFixed(2)}%`);

    // Wait a moment then check status
    await new Promise(resolve => setTimeout(resolve, 2000));

    const statusResponse = await axios.get(`${API_BASE}/api/evotester/${evoResponse.data.sessionId}/status`);
    console.log(`Status: ${statusResponse.data.status} (${statusResponse.data.progress}% complete)`);
    console.log(`Best Fitness: ${statusResponse.data.bestFitness}`);
    console.log(`Poor-Capital Mode Active: ${statusResponse.data.poorCapitalMode}`);

    console.log('\n🎉 Poor-Capital Mode Test Complete!');
    console.log('This demonstrates the three-ring approach:');
    console.log('1. ✅ Universe filtering with catalyst scoring');
    console.log('2. ✅ Position sizing with realistic constraints');
    console.log('3. ✅ EvoTester integration with Poor-Capital Mode');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

// Run the test
if (require.main === module) {
  testPoorCapitalMode();
}

module.exports = { testPoorCapitalMode };
