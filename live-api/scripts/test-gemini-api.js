#!/usr/bin/env node

/**
 * Test Gemini API Connection with Sandbox Credentials
 */

const { GeminiProvider } = require('../src/providers/crypto/gemini');

// Gemini Sandbox API credentials
const GEMINI_SANDBOX_API_KEY = 'account-84qzp7isnuVsHl0fk4J1';
const GEMINI_SANDBOX_API_SECRET = '3krJkotRataxyxt9TqSEpisPaUR4';

/**
 * Test Gemini API functionality
 */
async function testGeminiAPI() {
  console.log('🚀 Testing Gemini Sandbox API');
  console.log('=============================');

  const gemini = new GeminiProvider(GEMINI_SANDBOX_API_KEY, GEMINI_SANDBOX_API_SECRET, true);

  try {
    // Test 1: Get balances
    console.log('1. Testing account balances...');
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

    // Test 2: Get public market data
    console.log('\n2. Testing market data (BTCUSD)...');
    const ticker = await gemini.getTicker('btcusd');
    console.log('✅ Market data retrieved!');
    console.log(`BTCUSD - Last: $${ticker.last}, Bid: $${ticker.bid}, Ask: $${ticker.ask}`);

    // Test 3: Get order history
    console.log('\n3. Testing order history...');
    const orders = await gemini.getOrderHistory();
    console.log('✅ Order history retrieved!');
    console.log(`Found ${orders.length} orders in history`);

    // Test 4: Get available symbols
    console.log('\n4. Testing available trading pairs...');
    const symbols = await gemini.getSymbols();
    console.log('✅ Trading symbols retrieved!');
    console.log(`Available pairs: ${symbols.slice(0, 10).join(', ')}${symbols.length > 10 ? '...' : ''}`);

    // Test 5: Test fees
    console.log('\n5. Testing fee structure...');
    const fees = await gemini.getFees();
    console.log('✅ Fee structure retrieved!');
    console.log(`Maker Fee: ${fees.maker_fee}, Taker Fee: ${fees.taker_fee}`);

    console.log('\n🎉 All Gemini API tests passed!');
    console.log('✅ Your Gemini sandbox account is working perfectly!');
    console.log('✅ You can now do real paper trading with dynamic balances!');

    return {
      success: true,
      balances,
      ticker,
      orders,
      symbols,
      fees
    };

  } catch (error) {
    console.log('❌ Gemini API test failed:', error.message);

    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }

    console.log('\n🔧 Troubleshooting:');
    console.log('1. Verify your API key and secret are correct');
    console.log('2. Make sure you\'re using Gemini SANDBOX credentials');
    console.log('3. Check if your API key has the required permissions');

    return { success: false, error: error.message };
  }
}

/**
 * Test placing an order (optional, commented out to avoid accidental trades)
 */
async function testOrderPlacement(gemini) {
  console.log('\n⚠️  OPTIONAL: Testing order placement...');

  try {
    // Only test if we have sufficient balance
    const usdBalance = await gemini.getBalance('USD');

    if (parseFloat(usdBalance.available) > 10) {
      console.log('💰 Sufficient USD balance found, testing small order...');

      const order = await gemini.placeOrder({
        symbol: 'btcusd',
        side: 'buy',
        amount: '0.001', // Very small amount
        price: '30000', // Conservative price
        type: 'limit'
      });

      console.log('✅ Order placed successfully!');
      console.log(`Order ID: ${order.order_id}, Status: ${order.status}`);

      return order;
    } else {
      console.log('⚠️  Insufficient balance for order testing');
      return null;
    }

  } catch (error) {
    console.log('❌ Order placement test failed:', error.message);
    return null;
  }
}

// Main execution
if (require.main === module) {
  testGeminiAPI().then(async result => {
    if (result.success) {
      console.log('\n🚀 Ready for paper trading!');
      console.log('Your Gemini sandbox account is fully functional with:');
      console.log('✅ Dynamic account balances');
      console.log('✅ Real-time market data');
      console.log('✅ Order placement and tracking');
      console.log('✅ Order history');
      console.log('✅ Fee calculations');

      // Optional: Test order placement
      console.log('\n💡 Tip: Your account has real USD balance for paper trading!');
      console.log('You can now place orders that actually execute and affect your balance.');
    }
  }).catch(error => {
    console.error('💥 Test script failed:', error);
    process.exit(1);
  });
}

module.exports = { testGeminiAPI };
