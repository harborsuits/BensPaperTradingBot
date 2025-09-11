#!/usr/bin/env node

/**
 * Simple Gemini API Test - Check basic connectivity
 */

const axios = require('axios');

const SANDBOX_URL = 'https://api.sandbox.gemini.com';

/**
 * Test basic Gemini connectivity
 */
async function testBasicConnectivity() {
  console.log('🧪 Basic Gemini API Connectivity Test');
  console.log('======================================');

  try {
    // Test 1: Basic connectivity
    console.log('1. Testing basic connectivity...');
    const response = await axios.get(`${SANDBOX_URL}/v1/symbols`, {
      timeout: 10000
    });
    console.log('✅ Basic connectivity: OK');
    console.log(`Found ${response.data.length} trading pairs`);

    // Test 2: Market data
    console.log('\n2. Testing market data...');
    const ticker = await axios.get(`${SANDBOX_URL}/v1/pubticker/btcusd`, {
      timeout: 5000
    });
    console.log('✅ Market data: OK');
    console.log(`BTCUSD Last: $${ticker.data.last}`);

    // Test 3: Recent trades
    console.log('\n3. Testing recent trades...');
    const trades = await axios.get(`${SANDBOX_URL}/v1/trades/btcusd?limit_trades=5`, {
      timeout: 5000
    });
    console.log('✅ Recent trades: OK');
    console.log(`Last 5 trades for BTCUSD: ${trades.data.length} trades`);

    console.log('\n🎉 Gemini API is working for public endpoints!');
    console.log('The issue is with private endpoints authentication.');

    return true;

  } catch (error) {
    console.log('❌ Basic connectivity test failed:', error.message);

    if (error.response) {
      console.log(`Status: ${error.response.status}`);
      console.log('Response:', error.response.data);
    }

    return false;
  }
}

/**
 * Test if Gemini sandbox supports private endpoints at all
 */
async function testPrivateEndpointSupport() {
  console.log('\n🔐 Testing Private Endpoint Support');
  console.log('====================================');

  // Let's try to access a private endpoint without auth first
  try {
    const response = await axios.get(`${SANDBOX_URL}/v1/balances`, {
      timeout: 5000
    });

    console.log('✅ Private endpoint accessible without auth (unexpected!)');
    return true;

  } catch (error) {
    if (error.response?.status === 401) {
      console.log('✅ Private endpoint requires authentication (expected)');
      console.log('This means private endpoints DO exist, but need proper auth');
      return true;
    } else if (error.response?.status === 404) {
      console.log('❌ Private endpoints not available in sandbox');
      console.log('Gemini sandbox may not support private API endpoints');
      return false;
    } else {
      console.log(`❌ Unexpected error: ${error.response?.status || 'Unknown'}`);
      return false;
    }
  }
}

// Main execution
if (require.main === module) {
  testBasicConnectivity().then(success => {
    if (success) {
      return testPrivateEndpointSupport();
    }
  }).then(result => {
    console.log('\n📋 Next Steps:');
    console.log('==============');

    if (result) {
      console.log('✅ Gemini API is working');
      console.log('🔧 Need to fix authentication for private endpoints');
      console.log('💡 Check: API key permissions, signature generation, endpoint names');
    } else {
      console.log('❌ Gemini sandbox may not support private endpoints');
      console.log('💡 Consider using a different exchange for crypto paper trading');
      console.log('💡 Options: Binance Testnet, Kraken Demo, or custom simulator');
    }

    console.log('\n🔍 Debug Info:');
    console.log('- API Key: account-84qzp7isnuVsHl0fk4J1');
    console.log('- Using: HMAC-SHA384 signature');
    console.log('- Base64 encoded payload');
    console.log('- Sandbox URL: https://api.sandbox.gemini.com');
  }).catch(error => {
    console.error('💥 Test failed:', error);
  });
}

module.exports = { testBasicConnectivity, testPrivateEndpointSupport };
