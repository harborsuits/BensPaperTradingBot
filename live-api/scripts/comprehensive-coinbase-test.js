#!/usr/bin/env node

/**
 * Comprehensive Coinbase API Test Script
 * Tests all possible credential combinations and formats
 */

const axios = require('axios');
const crypto = require('crypto');

// User's new credential
const USER_CREDENTIAL = '118031293ae71c72ad2b327a27c14888.';
const SANDBOX_URL = 'https://api-public.sandbox.exchange.coinbase.com';
const PRODUCTION_URL = 'https://api.exchange.coinbase.com';

/**
 * Create Coinbase API authentication headers
 */
function createAuthHeaders(apiKey, apiSecret, method, path, body = '', passphrase = '') {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const message = timestamp + method + path + body;
  const signature = crypto
    .createHmac('sha256', apiSecret)
    .update(message)
    .digest('hex');

  return {
    'CB-ACCESS-KEY': apiKey,
    'CB-ACCESS-SIGN': signature,
    'CB-ACCESS-TIMESTAMP': timestamp,
    'CB-ACCESS-PASSPHRASE': passphrase,
    'Content-Type': 'application/json'
  };
}

/**
 * Test a specific credential combination
 */
async function testCredentials(name, apiKey, apiSecret, passphrase = '', isProduction = false) {
  const baseUrl = isProduction ? PRODUCTION_URL : SANDBOX_URL;
  const env = isProduction ? 'PRODUCTION' : 'SANDBOX';

  console.log(`\n🧪 Testing: ${name} (${env})`);
  console.log(`Key: ${apiKey.substring(0, 12)}...`);
  console.log(`Secret: ${apiSecret.substring(0, 12)}...`);
  if (passphrase) console.log(`Passphrase: ${passphrase.substring(0, 8)}...`);

  try {
    // Test accounts endpoint
    const accountsPath = '/accounts';
    const headers = createAuthHeaders(apiKey, apiSecret, 'GET', accountsPath, '', passphrase);

    const response = await axios.get(`${baseUrl}${accountsPath}`, {
      headers,
      timeout: 10000
    });

    console.log('✅ SUCCESS! Authentication works!');
    console.log(`Found ${response.data.length} accounts`);

    if (response.data.length > 0) {
      const currencies = response.data.map(acc => acc.currency).join(', ');
      console.log(`Currencies: ${currencies}`);
    }

    return {
      success: true,
      accounts: response.data,
      environment: env
    };

  } catch (error) {
    if (error.response && error.response.status === 401) {
      console.log('❌ 401 Unauthorized - Invalid credentials');
    } else if (error.response && error.response.status === 403) {
      console.log('❌ 403 Forbidden - Check API permissions');
    } else if (error.response) {
      console.log(`❌ HTTP ${error.response.status}: ${error.response.data.message || 'Unknown error'}`);
    } else {
      console.log(`❌ Network error: ${error.message}`);
    }
    return { success: false };
  }
}

/**
 * Comprehensive test of all possible combinations
 */
async function comprehensiveTest() {
  console.log('🔍 Comprehensive Coinbase API Test');
  console.log('===================================');

  const testCases = [
    // Sandbox tests
    {
      name: 'User cred as Key, Empty secret',
      key: USER_CREDENTIAL,
      secret: '',
      passphrase: ''
    },
    {
      name: 'User cred as Key, User cred as Secret',
      key: USER_CREDENTIAL,
      secret: USER_CREDENTIAL,
      passphrase: ''
    },
    {
      name: 'User cred as Key, User cred as Passphrase',
      key: USER_CREDENTIAL,
      secret: '',
      passphrase: USER_CREDENTIAL
    },
    // Try with known working partial credentials
    {
      name: 'Original Key, User cred as Secret',
      key: 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/',
      secret: USER_CREDENTIAL,
      passphrase: ''
    },
    {
      name: 'Original Key, User cred as Passphrase',
      key: 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/',
      secret: 'IfB//R/x6Ydn5eo/nXF7gw==',
      passphrase: USER_CREDENTIAL
    }
  ];

  for (const testCase of testCases) {
    const result = await testCredentials(
      testCase.name,
      testCase.key,
      testCase.secret,
      testCase.passphrase,
      false
    );

    if (result.success) {
      console.log('\n🎉 FOUND WORKING CREDENTIALS!');
      console.log(`Environment: ${result.environment}`);
      return result;
    }
  }

  console.log('\n❌ No working credential combinations found');
  return { success: false };
}

/**
 * Provide troubleshooting guidance
 */
function provideTroubleshooting() {
  console.log('\n🔧 Troubleshooting Guide:');
  console.log('=========================');
  console.log('1. Go to: https://www.coinbase.com/settings/api');
  console.log('2. Make sure you\'re creating SANDBOX API credentials (not production)');
  console.log('3. Required permissions:');
  console.log('   ✅ View account balances');
  console.log('   ✅ View orders and trades');
  console.log('   ✅ Place orders');
  console.log('4. Copy credentials exactly (no extra spaces)');
  console.log('5. Coinbase shows the API Secret only ONCE - save it immediately!');
  console.log('6. Common issues:');
  console.log('   • Using production credentials instead of sandbox');
  console.log('   • API key expired or deactivated');
  console.log('   • Missing required permissions');
  console.log('   • Extra spaces in copied credentials');
  console.log('\n📋 What you need to provide:');
  console.log('• API Key (32+ characters)');
  console.log('• API Secret (44+ characters, base64)');
  console.log('• Passphrase (if you set one during API creation)');
}

// Main execution
if (require.main === module) {
  comprehensiveTest().then(result => {
    if (!result.success) {
      provideTroubleshooting();
    }
  }).catch(error => {
    console.error('💥 Test failed:', error.message);
    provideTroubleshooting();
  });
}

module.exports = { comprehensiveTest };
