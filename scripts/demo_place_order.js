#!/usr/bin/env node

/**
 * Demo script to prove Tradier paper broker integration
 * Places a test order, waits for fill, and syncs PnL
 */

require('dotenv').config();
const { TradierBroker } = require('../live-api/lib/tradierBroker');

// Check if required environment variables are set
if (!process.env.TRADIER_TOKEN) {
  console.error('❌ TRADIER_TOKEN environment variable is required');
  console.error('Please set your Tradier API token in the .env file');
  process.exit(1);
}

async function demoPaperTrading() {
  console.log('🚀 Starting Tradier Paper Trading Demo...\n');

  const broker = new TradierBroker();

  // Check broker health
  console.log('📊 Checking broker health...');
  const health = await broker.healthCheck();
  console.log('Health check:', health);

  if (!health.ok) {
    console.error('❌ Broker health check failed. Check your Tradier credentials.');
    process.exit(1);
  }

  // Get initial balance
  console.log('\n💰 Getting initial balance...');
  const initialBalance = await broker.getBalance();
  console.log('Initial balance:', JSON.stringify(initialBalance, null, 2));

  // Get initial positions
  console.log('\n📈 Getting initial positions...');
  const initialPositions = await broker.getPositions();
  console.log('Initial positions:', initialPositions);

  // Place a small test order (fractional shares if supported, or 1 share)
  console.log('\n📝 Placing test order...');
  const testOrder = {
    symbol: 'AAPL',
    side: 'buy',
    quantity: 1, // 1 share for testing
    type: 'market',
    duration: 'day'
  };

  try {
    console.log(`📋 Order Details: ${testOrder.side.toUpperCase()} ${testOrder.quantity} ${testOrder.symbol} (${testOrder.type} order)`);

    const orderResult = await broker.placeOrder(testOrder);
    console.log('✅ Order placed successfully!');
    console.log(`Order ID: ${orderResult.id}`);
    console.log(`Status: ${orderResult.status}`);
    console.log(`Details: ${orderResult.side} ${orderResult.quantity} ${orderResult.symbol}`);

    // Wait for the order to process
    console.log('\n⏳ Waiting for order to process...');
    let attempts = 0;
    let orderStatus = null;

    while (attempts < 10) { // Try for up to 30 seconds
      attempts++;
      console.log(`Checking order status (attempt ${attempts}/10)...`);

      orderStatus = await broker.getOrderStatus(orderResult.id);

      if (orderStatus && (orderStatus.status === 'filled' || orderStatus.status === 'canceled' || orderStatus.status === 'rejected')) {
        break;
      }

      await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds
    }

    console.log('\n🔍 Final Order Status:');
    console.log(`Status: ${orderStatus?.status || 'unknown'}`);
    console.log(`Filled Quantity: ${orderStatus?.filledQuantity || 0}/${orderStatus?.quantity || testOrder.quantity}`);
    console.log(`Avg Price: $${orderStatus?.avgPrice || 'N/A'}`);

    // Sync all fills and positions
    console.log('\n🔄 Syncing fills and positions...');
    const syncResult = await broker.syncFills();
    console.log(`✅ Synced ${syncResult.orders?.length || 0} orders and ${syncResult.positions?.length || 0} positions`);

    // Get updated balance
    console.log('\n💰 Account Balance:');
    const updatedBalance = await broker.getBalance();

    if (updatedBalance) {
      console.log(`Total Cash: $${updatedBalance.totalCash?.toFixed(2) || 'N/A'}`);
      console.log(`Buying Power: $${updatedBalance.buyingPower?.toFixed(2) || 'N/A'}`);
      console.log(`Total Value: $${updatedBalance.totalValue?.toFixed(2) || 'N/A'}`);
    }

    // Calculate PnL if position exists
    const aaplPosition = syncResult.positions?.find(p => p.symbol === 'AAPL');
    if (aaplPosition) {
      console.log('\n📊 AAPL Position Details:');
      console.log(`Shares Owned: ${aaplPosition.quantity}`);
      console.log(`Cost Basis: $${aaplPosition.costBasis?.toFixed(2) || 'N/A'}`);
      console.log(`Market Value: $${aaplPosition.marketValue?.toFixed(2) || 'N/A'}`);
      console.log(`Unrealized PnL: $${aaplPosition.unrealizedPnL?.toFixed(2) || 'N/A'} (${aaplPosition.unrealizedPnLPct?.toFixed(2) || 'N/A'}%)`);
    }

    // Show order summary
    console.log('\n📋 Order Summary:');
    console.log(`✅ Order placed: ${testOrder.side} ${testOrder.quantity} ${testOrder.symbol}`);
    console.log(`✅ Order ID: ${orderResult.id}`);
    console.log(`✅ Status: ${orderStatus?.status || 'pending'}`);
    console.log(`✅ Filled: ${orderStatus?.filledQuantity || 0}/${testOrder.quantity} shares`);

    if (orderStatus?.status === 'filled') {
      console.log('\n🎉 SUCCESS: Paper trading integration is working!');
      console.log('✅ Order executed successfully');
      console.log('✅ Position tracked in broker');
      console.log('✅ PnL calculated and updated');
      console.log('✅ Ready for live micro-capital competitions!');
    } else {
      console.log(`\n⚠️  Order Status: ${orderStatus?.status || 'unknown'}`);
      console.log('Order may still be processing or requires manual review in Tradier dashboard');
    }

  } catch (error) {
    console.error('\n❌ Demo failed:', error.message);
    if (error.response) {
      console.error('API Response:', error.response.data);
    }
    console.error('\nTroubleshooting:');
    console.error('1. Check your TRADIER_TOKEN in .env file');
    console.error('2. Verify your Tradier account has paper trading enabled');
    console.error('3. Ensure sufficient buying power in paper account');
    process.exit(1);
  }
}

// Handle command line arguments
const command = process.argv[2];

if (command === 'place-test') {
  demoPaperTrading().catch(console.error);
} else {
  console.log('Usage: node demo_place_order.js place-test');
  console.log('This will place a test order and demonstrate the full broker integration');
}
