/**
 * Options Trading Integration Test
 * Demonstrates all the new options trading capabilities
 */

const axios = require('axios');

const API_BASE = 'http://localhost:4000';

async function testOptionsTrading() {
  console.log('🚀 Testing Options Trading Integration\n');

  try {
    // Test 1: Options Position Sizing
    console.log('📊 Test 1: Options Position Sizing');
    console.log('='.repeat(60));

    const positionResponse = await axios.post(`${API_BASE}/api/options/position-size`, {
      capital: 5000,
      optionType: 'vertical',
      underlyingPrice: 150.00,
      strike: 155.00, // OTM call
      expiry: '2025-12-20',
      ivRank: 0.7, // High IV
      expectedMove: 0.1,
      chainQuality: 0.8,
      frictionBudget: 0.12,
      conviction: 0.8
    });

    const sizing = positionResponse.data.sizing;
    console.log(`Strategy: ${positionResponse.data.symbol}`);
    console.log(`Contracts: ${sizing.contracts} (${sizing.shares} shares)`);
    console.log(`Premium: $${sizing.premium.toFixed(2)}`);
    console.log(`Max Loss: $${sizing.maxLoss.toFixed(2)}`);
    console.log(`Max Gain: $${sizing.maxGain.toFixed(2)}`);
    console.log(`Friction Ratio: ${(sizing.friction.ratio * 100).toFixed(1)}%`);
    console.log(`Can Execute: ${sizing.canExecute ? '✅' : '❌'} ${sizing.rejectionReason || ''}`);

    if (sizing.canExecute) {
      console.log(`Greeks: Δ${sizing.greeks.netDelta.toFixed(2)}, Γ${sizing.greeks.netGamma.toFixed(3)}, Θ${sizing.greeks.netTheta.toFixed(2)}, V${sizing.greeks.netVega.toFixed(2)}`);
      console.log(`Risk Metrics: Theta/Day $${sizing.riskMetrics.thetaDay.toFixed(2)}`);
    }

    // Test 2: Route Selection with Contextual Bandit
    console.log('\n🎯 Test 2: Route Selection');
    console.log('='.repeat(60));

    const routeResponse = await axios.post(`${API_BASE}/api/options/route-selection`, {
      ivRank: 0.7, // High IV favors verticals
      expectedMove: 0.08,
      trendStrength: 0.6,
      rvol: 1.2,
      chainQuality: 0.8,
      frictionHeadroom: 0.9,
      thetaHeadroom: 0.8,
      vegaHeadroom: 0.9,
      availableRoutes: ['vertical', 'long_call', 'long_put', 'equity']
    });

    const selection = routeResponse.data.selection;
    console.log(`Selected Route: ${selection.selectedRoute.type}`);
    console.log(`Confidence: ${(selection.confidence * 100).toFixed(1)}%`);
    console.log(`Rationale: ${selection.rationale}`);
    console.log(`Alternatives: ${selection.alternatives.map(r => r.type).join(', ')}`);

    // Test 3: Chain Analysis
    console.log('\n📈 Test 3: Options Chain Analysis');
    console.log('='.repeat(60));

    const chainResponse = await axios.post(`${API_BASE}/api/options/chain-analysis`, {
      underlying: 'AAPL',
      underlyingPrice: 150.00,
      calls: [
        { strike: 145, bid: 8.50, ask: 9.00, volume: 150, openInterest: 2500 },
        { strike: 150, bid: 5.20, ask: 5.60, volume: 200, openInterest: 1800 },
        { strike: 155, bid: 2.80, ask: 3.10, volume: 120, openInterest: 950 }
      ],
      puts: [
        { strike: 145, bid: 2.40, ask: 2.70, volume: 180, openInterest: 2100 },
        { strike: 150, bid: 4.80, ask: 5.20, volume: 160, openInterest: 1700 },
        { strike: 155, bid: 7.90, ask: 8.30, volume: 90, openInterest: 1200 }
      ]
    });

    const quality = chainResponse.data.quality;
    const validation = chainResponse.data.validation;
    const recommendations = chainResponse.data.recommendations;

    console.log(`Chain Quality: ${(quality.overall * 100).toFixed(1)}%`);
    console.log(`Spread Score: ${(quality.spreadScore * 100).toFixed(1)}%`);
    console.log(`Volume Score: ${(quality.volumeScore * 100).toFixed(1)}%`);
    console.log(`OI Score: ${(quality.oiScore * 100).toFixed(1)}%`);
    console.log(`Validation: ${validation.valid ? '✅' : '❌'} ${validation.reasons.join(', ')}`);
    console.log(`Recommendations:`);
    console.log(`  Verticals: ${recommendations.suitableForVerticals ? '✅' : '❌'}`);
    console.log(`  Long Options: ${recommendations.suitableForLongOptions ? '✅' : '❌'}`);
    console.log(`  Risk Level: ${recommendations.riskLevel.toUpperCase()}`);

    // Test 4: Update Bandit Model
    console.log('\n🧠 Test 4: Bandit Model Update');
    console.log('='.repeat(60));

    const updateResponse = await axios.post(`${API_BASE}/api/options/update-bandit`, {
      route: selection.selectedRoute.type,
      context: routeResponse.data.context,
      outcome: {
        pnl: 45.50,
        friction: 0.08,
        drawdown: 0.02,
        duration: 5
      }
    });

    console.log(`Update Result: ${updateResponse.data.success ? '✅' : '❌'}`);
    console.log(`Route: ${updateResponse.data.route}`);

    // Test 5: Route Statistics
    console.log('\n📊 Test 5: Route Performance Statistics');
    console.log('='.repeat(60));

    const statsResponse = await axios.get(`${API_BASE}/api/options/route-stats`);
    const stats = statsResponse.data.statistics;

    console.log(`Total Routes Tracked: ${statsResponse.data.totalRoutes}`);
    Object.entries(stats).forEach(([route, performance]) => {
      console.log(`${route}:`);
      console.log(`  Trades: ${performance.totalTrades}`);
      console.log(`  Avg P&L: $${performance.avgPnL.toFixed(2)}`);
      console.log(`  Win Rate: ${(performance.winRate * 100).toFixed(1)}%`);
      console.log(`  Avg Friction: ${(performance.avgFriction * 100).toFixed(1)}%`);
    });

    console.log('\n🎉 Options Trading Integration Complete!');
    console.log('='.repeat(60));
    console.log('✅ Implemented Features:');
    console.log('• IV-aware position sizing');
    console.log('• Contextual bandit route selection');
    console.log('• Options chain quality analysis');
    console.log('• Greeks-based risk management');
    console.log('• Friction budget controls');
    console.log('• Theta governor protection');
    console.log('• Expected move validation');
    console.log('• Performance tracking and learning');

    console.log('\n💡 Key Benefits for Small Accounts:');
    console.log(`• Defined risk: Max loss of $${sizing.maxLoss?.toFixed(2) || 'N/A'} vs unlimited equity risk`);
    console.log(`• Leverage efficiency: ${sizing.contracts} contracts control ${sizing.shares} shares`);
    console.log(`• Theta income: $${(sizing.riskMetrics?.thetaDay || 0).toFixed(2)} daily decay collection`);
    console.log(`• Friction control: ${(sizing.friction?.ratio || 0 * 100).toFixed(1)}% vs typical 15-25%`);
    console.log(`• Adaptive routing: ${selection.selectedRoute.type} selected for ${(selection.confidence * 100).toFixed(1)}% confidence`);

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

// Run the test
if (require.main === module) {
  testOptionsTrading();
}

module.exports = { testOptionsTrading };
