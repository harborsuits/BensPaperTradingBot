#!/usr/bin/env node

/**
 * Simple Coinbase Sandbox Funding Script
 *
 * This script funds a Coinbase sandbox account with USD for paper trading.
 * Uses direct API calls without TypeScript dependencies.
 */

const axios = require('axios');
const crypto = require('crypto');

// Coinbase Sandbox API credentials
const COINBASE_SANDBOX_API_KEY = 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/';
const COINBASE_SANDBOX_API_SECRET = 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/IfB//R/x6Ydn5eo/nXF7gw==';
const COINBASE_SANDBOX_URL = 'https://api-public.sandbox.exchange.coinbase.com';

// Default funding parameters
const DEFAULT_AMOUNT = 10000;
const DEFAULT_CURRENCY = 'USD';

// Get command line arguments
const amount = process.argv[2] ? parseFloat(process.argv[2]) : DEFAULT_AMOUNT;
const currency = process.argv[3] || DEFAULT_CURRENCY;

/**
 * Create Coinbase API authentication headers
 */
function createAuthHeaders(method, path, body = '') {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const message = timestamp + method + path + body;
  const signature = crypto
    .createHmac('sha256', COINBASE_SANDBOX_API_SECRET)
    .update(message)
    .digest('hex');

  return {
    'CB-ACCESS-KEY': COINBASE_SANDBOX_API_KEY,
    'CB-ACCESS-SIGN': signature,
    'CB-ACCESS-TIMESTAMP': timestamp,
    'CB-ACCESS-PASSPHRASE': '', // Not required for sandbox
    'Content-Type': 'application/json'
  };
}

/**
 * Get account balances
 */
async function getBalances() {
  try {
    const path = '/accounts';
    const headers = createAuthHeaders('GET', path);

    const response = await axios.get(`${COINBASE_SANDBOX_URL}${path}`, {
      headers,
      timeout: 5000
    });

    const result = {};

    for (const account of response.data) {
      if (account.currency) {
        result[account.currency] = {
          asset: account.currency,
          free: parseFloat(account.available || '0'),
          locked: parseFloat(account.hold || '0'),
          total: parseFloat(account.balance || '0')
        };
      }
    }

    return result;
  } catch (error) {
    console.error('Error getting balances:', error.message);
    return {};
  }
}

/**
 * Fund sandbox account
 */
async function fundSandboxAccount(currency, amount) {
  try {
    const path = '/accounts/sandbox-funding';
    const body = JSON.stringify({
      currency,
      amount: amount.toString()
    });

    const headers = createAuthHeaders('POST', path, body);

    const response = await axios.post(`${COINBASE_SANDBOX_URL}${path}`, body, {
      headers,
      timeout: 10000
    });

    return response.data;
  } catch (error) {
    console.error('Error funding account:', error.message);
    if (error.response) {
      console.error('API Response:', error.response.data);
    }
    throw error;
  }
}

/**
 * Check API health
 */
async function checkApiHealth() {
  try {
    const path = '/time';
    const headers = createAuthHeaders('GET', path);

    await axios.get(`${COINBASE_SANDBOX_URL}${path}`, {
      headers,
      timeout: 3000
    });
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Main funding function
 */
async function fundAccount() {
  console.log('🚀 Coinbase Sandbox Account Funding Script');
  console.log('==========================================');
  console.log(`Amount: $${amount.toLocaleString()} ${currency}`);
  console.log('');

  try {
    console.log('📡 Connecting to Coinbase Sandbox API...');

    // Check API health
    const isHealthy = await checkApiHealth();
    if (!isHealthy) {
      throw new Error('Coinbase API is not responding');
    }
    console.log('✅ Coinbase API connection successful');

    // Get current balances
    console.log('📊 Checking current balances...');
    const balancesBefore = await getBalances();
    const usdBalance = balancesBefore[currency];
    console.log(`💰 Current ${currency} balance: $${(usdBalance?.total || 0).toFixed(2)}`);

    // Fund the account
    console.log(`💸 Funding account with $${amount.toLocaleString()} ${currency}...`);
    const fundingResult = await fundSandboxAccount(currency, amount);

    console.log('✅ Funding successful!');
    console.log('📄 Transaction details:', JSON.stringify(fundingResult, null, 2));

    // Wait for transaction to process
    console.log('⏳ Waiting for transaction to process...');
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Get updated balances
    console.log('📊 Checking updated balances...');
    const balancesAfter = await getBalances();
    const usdBalanceAfter = balancesAfter[currency];
    console.log(`💰 Updated ${currency} balance: $${(usdBalanceAfter?.total || 0).toFixed(2)}`);

    // Calculate difference
    const difference = (usdBalanceAfter?.total || 0) - (usdBalance?.total || 0);
    console.log(`📈 Balance change: +$${difference.toFixed(2)} ${currency}`);

    console.log('');
    console.log('🎉 Sandbox account funding completed successfully!');
    console.log('💡 You can now use this account for paper trading on Coinbase Sandbox.');

  } catch (error) {
    console.error('❌ Funding failed:', error.message);

    console.log('');
    console.log('🔍 Troubleshooting tips:');
    console.log('1. Verify your Coinbase Sandbox API credentials are correct');
    console.log('2. Make sure you\'re using the sandbox environment');
    console.log('3. Check your internet connection');
    console.log('4. Verify the API key has the necessary permissions');
    console.log('5. Try a smaller amount first (e.g., $1000)');

    process.exit(1);
  }
}

// Run the script
if (require.main === module) {
  fundAccount().catch(error => {
    console.error('💥 Script execution failed:', error);
    process.exit(1);
  });
}

module.exports = { fundAccount };
