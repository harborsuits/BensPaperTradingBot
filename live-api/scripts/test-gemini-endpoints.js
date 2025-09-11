#!/usr/bin/env node

/**
 * Test Gemini API Endpoints to find the correct ones
 */

const axios = require('axios');
const crypto = require('crypto');

// Gemini Sandbox API credentials
const GEMINI_SANDBOX_API_KEY = 'account-84qzp7isnuVsHl0fk4J1';
const GEMINI_SANDBOX_API_SECRET = '3krJkotRataxyxt9TqSEpisPaUR4';
const SANDBOX_URL = 'https://api.sandbox.gemini.com';

/**
 * Generate HMAC SHA384 signature for Gemini API authentication
 */
function generateSignature(payload) {
  const secret = Buffer.from(GEMINI_SANDBOX_API_SECRET, 'base64');
  const message = JSON.stringify(payload);
  return crypto
    .createHmac('sha384', secret)
    .update(message)
    .digest('hex');
}

/**
 * Test different endpoints
 */
async function testEndpoints() {
  console.log('🔍 Testing Gemini API Endpoints');
  console.log('================================');

  // First, test public endpoints that don't require auth
  console.log('\n1. Testing public endpoints...');

  const publicEndpoints = [
    '/v1/symbols',
    '/v1/pubticker/btcusd',
    '/v1/trades/btcusd'
  ];

  for (const endpoint of publicEndpoints) {
    try {
      const response = await axios.get(`${SANDBOX_URL}${endpoint}`, {
        timeout: 5000
      });
      console.log(`✅ ${endpoint}: OK`);
    } catch (error) {
      console.log(`❌ ${endpoint}: ${error.response?.status || 'ERROR'}`);
    }
  }

  // Test private endpoints with auth
  console.log('\n2. Testing private endpoints with auth...');

  const privateEndpoints = [
    '/v1/balances',
    '/v1/balance',
    '/v1/account',
    '/v1/accounts',
    '/v1/notionalvolume',
    '/v1/tradevolume',
    '/v1/availablebalances',
    '/v1/mytrades',
    '/v1/orders',
    '/v1/order/new',
    '/v1/clearing/new',
    '/v1/deposit/new',
    '/v1/withdraw/new'
  ];

  for (const endpoint of privateEndpoints) {
    try {
      const payload = {
        request: endpoint,
        nonce: Date.now()
      };

      const response = await axios.get(`${SANDBOX_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'text/plain',
          'Content-Length': '0',
          'X-GEMINI-APIKEY': GEMINI_SANDBOX_API_KEY,
          'X-GEMINI-PAYLOAD': Buffer.from(JSON.stringify(payload)).toString('base64'),
          'X-GEMINI-SIGNATURE': generateSignature(payload),
          'Cache-Control': 'no-cache'
        },
        timeout: 5000
      });

      console.log(`✅ ${endpoint}: OK`);
      if (Array.isArray(response.data) && response.data.length > 0) {
        console.log(`   Sample: ${JSON.stringify(response.data[0]).substring(0, 100)}...`);
      }
    } catch (error) {
      const status = error.response?.status || 'ERROR';
      const message = error.response?.data?.message || error.message;
      console.log(`❌ ${endpoint}: ${status} - ${message}`);
    }
  }

  // Test different API versions
  console.log('\n3. Testing different API versions...');
  const versions = ['v1', 'v2'];

  for (const version of versions) {
    try {
      const response = await axios.get(`${SANDBOX_URL}/${version}/symbols`, {
        timeout: 5000
      });
      console.log(`✅ API ${version}: OK`);
      console.log(`   Symbols: ${response.data.slice(0, 5).join(', ')}...`);
    } catch (error) {
      console.log(`❌ API ${version}: ${error.response?.status || 'ERROR'}`);
    }
  }
}

// Main execution
if (require.main === module) {
  testEndpoints().then(() => {
    console.log('\n📋 Summary:');
    console.log('===========');
    console.log('The correct Gemini API endpoints need to be identified.');
    console.log('Common endpoints include:');
    console.log('- /v1/balances (or similar)');
    console.log('- /v1/account');
    console.log('- /v1/orders');
    console.log('- /v1/trades');
  }).catch(error => {
    console.error('💥 Test failed:', error);
  });
}

module.exports = { testEndpoints };
