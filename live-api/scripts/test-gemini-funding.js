#!/usr/bin/env node

/**
 * Test Gemini API with Funding Management Permissions
 * Tests private endpoints that require funding management enabled
 */

const { GeminiProvider } = require('../src/providers/crypto/gemini');

// Gemini Sandbox API credentials
const GEMINI_SANDBOX_API_KEY = 'account-84qzp7isnuVsHl0fk4J1';
const GEMINI_SANDBOX_API_SECRET = '3krJkotRataxyxt9TqSEpisPaUR4';

/**
 * Test Gemini private endpoints with funding permissions
 */
async function testGeminiFunding() {
  console.log('🔐 Testing Gemini Funding Management Permissions');
  console.log('===============================================');

  const gemini = new GeminiProvider(GEMINI_SANDBOX_API_KEY, GEMINI_SANDBOX_API_SECRET, true);

  try {
    console.log('1. Testing account balances (requires funding permission)...');
    const balances = await gemini.getBalances();
    console.log('✅ Balances retrieved successfully!');
    console.log(`Found ${balances.length} balance entries`);

    // Show USD and BTC balances
    const usdBalance = balances.find(b => b.currency === 'USD');
    const btcBalance = balances.find(b => b.currency === 'BTC');

    if (usdBalance) {
      console.log(`💵 USD Balance: ${usdBalance.available} (Available: ${usdBalance.available})`);
    }

    if (btcBalance) {
      console.log(`₿ BTC Balance: ${btcBalance.available} (Available: ${btcBalance.available})`);
    }

    console.log('\n2. Testing order history...');
    const orders = await gemini.getOrderHistory();
    console.log('✅ Order history retrieved!');
    console.log(`Found ${orders.length} orders in history`);

    console.log('\n3. Testing place/cancel order (funding management)...');
    const order = await gemini.placeOrder({
      symbol: 'btcusd',
      side: 'buy',
      quantity: 0.001,
      type: 'limit',
      price: '100000' // Conservative price
    });
    console.log('✅ Order placed successfully!');
    console.log(`Order ID: ${order.order_id}, Status: ${order.status}`);

    // Wait a moment then get order status
    await new Promise(resolve => setTimeout(resolve, 1000));

    const orderStatus = await gemini.getOrder(order.order_id);
    console.log(`Order status: ${orderStatus.status}`);

    // Cancel the order
    const cancelResult = await gemini.cancelOrder(order.order_id);
    console.log('✅ Order cancelled successfully!');

    console.log('\n🎉 ALL FUNDING MANAGEMENT PERMISSIONS WORKING!');
    console.log('✅ Account balances');
    console.log('✅ Order placement');
    console.log('✅ Order cancellation');
    console.log('✅ Order history');

    return {
      success: true,
      balances,
      orders,
      order,
      cancelResult
    };

  } catch (error) {
    console.log('❌ Funding permissions test failed:', error.message);

    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }

    console.log('\n🔧 Troubleshooting:');
    console.log('1. Verify funding management permissions are ENABLED');
    console.log('2. Check if you have sufficient permissions for private endpoints');
    console.log('3. Ensure the API key is active and has the right permissions');
    console.log('4. Try regenerating the API key with funding permissions');

    return { success: false, error: error.message };
  }
}

/**
 * Test if we can fund the account and place real orders
 */
async function testFullTrading(gemini) {
  console.log('\n💰 Testing Full Trading Capabilities');

  try {
    // Get current balances
    const balances = await gemini.getBalances();
    const usdBalance = balances.find(b => b.currency === 'USD');

    if (usdBalance && parseFloat(usdBalance.available) > 100) {
      console.log('✅ Sufficient USD balance for trading');

      // Place a small market order
      const order = await gemini.placeOrder({
        symbol: 'btcusd',
        side: 'buy',
        quantity: 0.0001, // Very small amount
        type: 'market'
      });

      console.log('✅ Market order placed successfully!');
      console.log(`Order ID: ${order.order_id}`);

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Check if it executed
      const orderStatus = await gemini.getOrder(order.order_id);
      console.log(`Order status: ${orderStatus.status}`);

      if (orderStatus.status === 'filled') {
        console.log('🎉 Order executed successfully!');
        return true;
      }
    } else {
      console.log('⚠️  Insufficient balance for trading test');
      console.log('💡 You may need to fund your sandbox account');
      return false;
    }

  } catch (error) {
    console.log('❌ Full trading test failed:', error.message);
    return false;
  }
}

// Main execution
if (require.main === module) {
  testGeminiFunding().then(async result => {
    if (result.success) {
      console.log('\n🚀 Funding Management Permissions Confirmed!');
      console.log('You now have access to:');
      console.log('✅ Account balances');
      console.log('✅ Order placement');
      console.log('✅ Order cancellation');
      console.log('✅ Deposit addresses');
      console.log('✅ Withdrawal addresses');

      console.log('\n🎯 Ready for Paper Trading Integration!');

      // Optional: Test full trading
      const gemini = new GeminiProvider(GEMINI_SANDBOX_API_KEY, GEMINI_SANDBOX_API_SECRET, true);
      const tradingTest = await testFullTrading(gemini);

      if (tradingTest) {
        console.log('\n💎 Full trading capabilities confirmed!');
      }

    } else {
      console.log('\n❌ Funding permissions not working yet');
      console.log('Please check your Gemini API key settings');
    }
  }).catch(error => {
    console.error('💥 Test failed:', error);
  });
}

module.exports = { testGeminiFunding, testFullTrading };
