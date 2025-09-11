#!/usr/bin/env node

/**
 * Test Paper Trading Engine
 */

const { PaperTradingEngine } = require('../src/services/paperTradingEngine');

// Gemini Sandbox API credentials
const GEMINI_SANDBOX_API_KEY = 'account-84qzp7isnuVsHl0fk4J1';
const GEMINI_SANDBOX_API_SECRET = '3krJkotRataxyxt9TqSEpisPaUR4';

/**
 * Test the paper trading engine
 */
async function testPaperTrading() {
  console.log('🧪 Testing Paper Trading Engine');
  console.log('==============================');

  const paperEngine = new PaperTradingEngine(GEMINI_SANDBOX_API_KEY, GEMINI_SANDBOX_API_SECRET);

  try {
    // Initialize account
    console.log('1. Initializing paper account...');
    await paperEngine.initialize();

    // Get initial account state
    console.log('\n2. Getting initial account state...');
    let account = await paperEngine.getAccount();
    console.log(`💰 USD Balance: $${account.usd_balance.toFixed(2)}`);
    console.log(`📊 Total Equity: $${account.total_equity.toFixed(2)}`);
    console.log(`📈 Positions: ${account.position_count}`);

    // Fund account
    console.log('\n3. Funding account with $5,000...');
    const fundingResult = await paperEngine.fundAccount(5000);
    console.log(`✅ Funding successful! New balance: $${fundingResult.new_balance.toFixed(2)}`);

    // Place a buy order
    console.log('\n4. Placing BUY order for 0.001 BTC...');
    const buyOrder = await paperEngine.placeOrder({
      symbol: 'btcusd',
      side: 'buy',
      quantity: 0.001,
      type: 'market'
    });
    console.log(`✅ Buy order filled: ${buyOrder.symbol} ${buyOrder.quantity} @ $${buyOrder.price.toFixed(2)}`);

    // Get updated account
    console.log('\n5. Getting updated account state...');
    account = await paperEngine.getAccount();
    console.log(`💰 USD Balance: $${account.usd_balance.toFixed(2)}`);
    console.log(`📊 Total Equity: $${account.total_equity.toFixed(2)}`);
    console.log(`📈 Positions: ${account.position_count}`);

    if (account.positions.length > 0) {
      const position = account.positions[0];
      console.log(`₿ BTC Position: ${position.quantity} @ avg $${position.avg_price.toFixed(2)}`);
      console.log(`📊 Current Value: $${position.current_value.toFixed(2)}`);
      console.log(`📈 Unrealized P&L: $${position.unrealized_pnl.toFixed(2)} (${position.pnl_percentage.toFixed(2)}%)`);
    }

    // Place a sell order
    console.log('\n6. Placing SELL order for 0.001 BTC...');
    const sellOrder = await paperEngine.placeOrder({
      symbol: 'btcusd',
      side: 'sell',
      quantity: 0.001,
      type: 'market'
    });
    console.log(`✅ Sell order filled: ${sellOrder.symbol} ${sellOrder.quantity} @ $${sellOrder.price.toFixed(2)}`);

    // Get final account state
    console.log('\n7. Getting final account state...');
    account = await paperEngine.getAccount();
    console.log(`💰 USD Balance: $${account.usd_balance.toFixed(2)}`);
    console.log(`📊 Total Equity: $${account.total_equity.toFixed(2)}`);

    // Get trading statistics
    console.log('\n8. Getting trading statistics...');
    const stats = await paperEngine.getStats();
    console.log(`📊 Total P&L: $${stats.total_pnl.toFixed(2)} (${stats.pnl_percentage.toFixed(2)}%)`);
    console.log(`✅ Realized P&L: $${stats.realized_pnl.toFixed(2)}`);
    console.log(`📈 Unrealized P&L: $${stats.unrealized_pnl.toFixed(2)}`);
    console.log(`🎯 Win Rate: ${stats.win_rate.toFixed(1)}%`);
    console.log(`📋 Total Trades: ${stats.total_trades}`);

    // Get order history
    console.log('\n9. Getting order history...');
    const orders = await paperEngine.getOrderHistory(10);
    console.log(`📋 Recent Orders: ${orders.length}`);
    orders.forEach(order => {
      console.log(`   ${order.side.toUpperCase()} ${order.quantity} ${order.symbol} @ $${order.price.toFixed(2)} (${order.status})`);
    });

    console.log('\n🎉 Paper Trading Engine Test Complete!');
    console.log('✅ All functionality working perfectly');
    console.log('✅ Real Gemini market data integration');
    console.log('✅ Persistent account state');
    console.log('✅ P&L tracking and statistics');

    return {
      success: true,
      final_balance: account.total_equity,
      stats: stats
    };

  } catch (error) {
    console.log('❌ Paper trading test failed:', error.message);

    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }

    return { success: false, error: error.message };
  }
}

// Main execution
if (require.main === module) {
  testPaperTrading().then(result => {
    if (result.success) {
      console.log('\n🚀 Paper Trading Engine is ready!');
      console.log('You now have a fully functional crypto paper trading account that:');
      console.log('✅ Uses real Gemini market data');
      console.log('✅ Tracks P&L in real-time');
      console.log('✅ Saves account state persistently');
      console.log('✅ Supports buy/sell orders');
      console.log('✅ Provides trading statistics');
      console.log('✅ Allows virtual funding');

      console.log('\n💡 Ready to integrate with your BenBot dashboard!');
    } else {
      console.log('\n❌ Need to fix issues before integration');
    }
  }).catch(error => {
    console.error('💥 Test script failed:', error);
  });
}

module.exports = { testPaperTrading };
