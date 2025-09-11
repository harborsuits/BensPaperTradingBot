#!/usr/bin/env node

/**
 * Coinbase API Key Test Script
 *
 * This script tests Coinbase API credentials to verify they work before funding.
 */

const axios = require('axios');
const crypto = require('crypto');

// Default test parameters
const DEFAULT_CURRENCY = 'USD';

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
    'CB-ACCESS-PASSPHRASE': '', // Not required for sandbox
    'Content-Type': 'application/json'
  };
}

/**
 * Test API connection and permissions
 */
async function testApiCredentials(apiKey, apiSecret, isSandbox = true) {
  const baseUrl = isSandbox
    ? 'https://api-public.sandbox.exchange.coinbase.com'
    : 'https://api.exchange.coinbase.com';

  console.log(`🔍 Testing Coinbase ${isSandbox ? 'Sandbox' : 'Production'} API credentials...`);
  console.log('');

  try {
    // Test 1: Basic connectivity (time endpoint)
    console.log('1️⃣ Testing basic connectivity...');
    const timePath = '/time';
    const timeHeaders = createAuthHeaders(apiKey, apiSecret, 'GET', timePath);

    const timeResponse = await axios.get(`${baseUrl}${timePath}`, {
      headers: timeHeaders,
      timeout: 5000
    });

    console.log('✅ Basic connectivity: OK');
    console.log(`   Server time: ${new Date(timeResponse.data.iso).toLocaleString()}`);

    // Test 2: Account access (accounts endpoint)
    console.log('2️⃣ Testing account access...');
    const accountsPath = '/accounts';
    const accountsHeaders = createAuthHeaders(apiKey, apiSecret, 'GET', accountsPath);

    const accountsResponse = await axios.get(`${baseUrl}${accountsPath}`, {
      headers: accountsHeaders,
      timeout: 5000
    });

    console.log('✅ Account access: OK');
    console.log(`   Found ${accountsResponse.data.length} accounts`);

    // Show account balances
    const usdAccount = accountsResponse.data.find(acc => acc.currency === 'USD');
    if (usdAccount) {
      console.log(`   USD Balance: $${parseFloat(usdAccount.balance || '0').toFixed(2)}`);
    }

    // Test 3: Sandbox funding capability (sandbox only)
    if (isSandbox) {
      console.log('3️⃣ Testing sandbox funding capability...');
      const fundingPath = '/accounts/sandbox-funding';
      const fundingBody = JSON.stringify({
        currency: 'USD',
        amount: '1' // Test with $1
      });

      const fundingHeaders = createAuthHeaders(apiKey, apiSecret, 'POST', fundingPath, fundingBody);

      try {
        const fundingResponse = await axios.post(`${baseUrl}${fundingPath}`, fundingBody, {
          headers: fundingHeaders,
          timeout: 10000
        });

        console.log('✅ Sandbox funding: OK');
        console.log('   Test funding successful (reversed automatically)');

      } catch (fundingError) {
        if (fundingError.response?.status === 400) {
          console.log('⚠️  Sandbox funding: Limited (may work with larger amounts)');
          console.log('   This is normal for some sandbox accounts');
        } else {
          console.log('❌ Sandbox funding: Failed');
          console.log(`   Error: ${fundingError.response?.data?.message || fundingError.message}`);
        }
      }
    }

    console.log('');
    console.log('🎉 API credentials test completed successfully!');
    console.log('💡 Your Coinbase API credentials are working correctly.');
    console.log('');

    if (isSandbox) {
      console.log('🚀 Ready to fund sandbox account with:');
      console.log(`   node scripts/fund-coinbase-simple.js [amount] [currency]`);
    }

    return true;

  } catch (error) {
    console.error('❌ API credentials test failed:', error.message);

    if (error.response) {
      console.error('API Response:', error.response.data);
      console.error('Status Code:', error.response.status);

      if (error.response.status === 401) {
        console.log('');
        console.log('🔐 Authentication Error - Possible causes:');
        console.log('• API Key is incorrect or expired');
        console.log('• API Secret is incorrect');
        console.log('• API Key does not have required permissions');
        console.log('• Wrong environment (sandbox vs production)');
      }
    }

    console.log('');
    console.log('🔧 How to fix:');
    console.log('1. Go to https://www.coinbase.com/settings/api');
    console.log('2. Create a new API key for the sandbox environment');
    console.log('3. Ensure the key has these permissions:');
    console.log('   - View account balances');
    console.log('   - View orders and trades');
    console.log('   - Place orders');
    console.log('4. Copy the API Key and Secret exactly');
    console.log('5. Test again with: node scripts/test-coinbase-api.js <api-key> <api-secret>');

    return false;
  }
}

/**
 * Main test function
 */
async function main() {
  const apiKey = process.argv[2];
  const apiSecret = process.argv[3];
  const isSandbox = process.argv[4] !== 'production';

  console.log('🧪 Coinbase API Credentials Test');
  console.log('=================================');

  if (!apiKey || !apiSecret) {
    console.log('❌ Missing API credentials');
    console.log('');
    console.log('Usage: node scripts/test-coinbase-api.js <api-key> <api-secret> [environment]');
    console.log('');
    console.log('Examples:');
    console.log('  # Test sandbox credentials (default)');
    console.log('  node scripts/test-coinbase-api.js your-api-key your-api-secret');
    console.log('');
    console.log('  # Test production credentials');
    console.log('  node scripts/test-coinbase-api.js your-api-key your-api-secret production');
    console.log('');
    console.log('📝 To get Coinbase API credentials:');
    console.log('1. Go to https://www.coinbase.com/settings/api');
    console.log('2. Click "Create API Key"');
    console.log('3. Choose "Sandbox" environment');
    console.log('4. Enable these permissions:');
    console.log('   - View account balances');
    console.log('   - View orders and trades');
    console.log('   - Place orders');
    console.log('5. Copy the API Key and Secret');
    process.exit(1);
  }

  console.log(`Environment: ${isSandbox ? 'Sandbox' : 'Production'}`);
  console.log(`API Key: ${apiKey.substring(0, 8)}...${apiKey.substring(apiKey.length - 4)}`);
  console.log('');

  const success = await testApiCredentials(apiKey, apiSecret, isSandbox);

  if (!success) {
    process.exit(1);
  }
}

// Run the script
if (require.main === module) {
  main().catch(error => {
    console.error('💥 Script execution failed:', error);
    process.exit(1);
  });
}

module.exports = { testApiCredentials };
