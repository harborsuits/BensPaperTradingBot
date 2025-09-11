#!/usr/bin/env node

/**
 * Debug Coinbase API Script
 * Helps debug API key format and authentication issues
 */

const axios = require('axios');
const crypto = require('crypto');

// Test with different API key formats
const testCredentials = [
  {
    name: 'Original format',
    key: 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/',
    secret: 'IfB//R/x6Ydn5eo/nXF7gw=='
  },
  {
    name: 'User provided (split)',
    key: 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/',
    secret: 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/IfB//R/x6Ydn5eo/nXF7gw=='
  }
];

const COINBASE_SANDBOX_URL = 'https://api-public.sandbox.exchange.coinbase.com';

/**
 * Create Coinbase API authentication headers
 */
function createAuthHeaders(apiKey, apiSecret, method, path, body = '') {
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
    'CB-ACCESS-PASSPHRASE': '',
    'Content-Type': 'application/json'
  };
}

/**
 * Test a single credential set
 */
async function testCredentialSet(name, apiKey, apiSecret) {
  console.log(`\n🧪 Testing: ${name}`);
  console.log(`API Key: ${apiKey}`);
  console.log(`API Secret: ${apiSecret.substring(0, 20)}...`);

  try {
    // Test time endpoint (doesn't require authentication)
    console.log('Testing time endpoint (no auth)...');
    const timeResponse = await axios.get(`${COINBASE_SANDBOX_URL}/time`);
    console.log('✅ Time endpoint: OK');

    // Test accounts endpoint (requires authentication)
    console.log('Testing accounts endpoint (with auth)...');
    const accountsPath = '/accounts';
    const headers = createAuthHeaders(apiKey, apiSecret, 'GET', accountsPath);

    const accountsResponse = await axios.get(`${COINBASE_SANDBOX_URL}${accountsPath}`, {
      headers,
      timeout: 5000
    });

    console.log('✅ Accounts endpoint: OK');
    console.log(`Found ${accountsResponse.data.length} accounts`);

    return { success: true, accounts: accountsResponse.data };

  } catch (error) {
    console.log('❌ Failed:', error.message);
    if (error.response) {
      console.log('Status:', error.response.status);
      console.log('Response:', error.response.data);
    }
    return { success: false, error: error.message };
  }
}

/**
 * Main debug function
 */
async function debugCredentials() {
  console.log('🔧 Coinbase API Debug Tool');
  console.log('===========================');

  for (const cred of testCredentials) {
    const result = await testCredentialSet(cred.name, cred.key, cred.secret);
    if (result.success) {
      console.log(`🎉 SUCCESS: ${cred.name} works!`);
      console.log('Accounts:', result.accounts.map(acc => `${acc.currency}: ${acc.balance}`).join(', '));
      break;
    } else {
      console.log(`❌ FAILED: ${cred.name} doesn't work`);
    }
  }

  console.log('\n📝 Troubleshooting:');
  console.log('1. Make sure you\'re using SANDBOX credentials (not production)');
  console.log('2. Verify the API key and secret are copied exactly from Coinbase');
  console.log('3. Check that the API key has these permissions:');
  console.log('   - View account balances');
  console.log('   - View orders and trades');
  console.log('   - Place orders');
  console.log('4. Try creating a new API key if the current one is expired');
}

// Run the debug script
if (require.main === module) {
  debugCredentials().catch(error => {
    console.error('💥 Debug script failed:', error);
    process.exit(1);
  });
}

module.exports = { debugCredentials };
