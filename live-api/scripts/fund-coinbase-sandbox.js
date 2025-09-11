#!/usr/bin/env node

/**
 * Coinbase Sandbox Funding Script
 *
 * This script funds a Coinbase sandbox account with USD for paper trading.
 * Usage: node scripts/fund-coinbase-sandbox.js [amount] [currency]
 */

const { CoinbaseCryptoProvider } = require('../src/providers/crypto/coinbase');

// Default funding parameters
const DEFAULT_AMOUNT = 10000;
const DEFAULT_CURRENCY = 'USD';

// Get command line arguments
const amount = process.argv[2] ? parseFloat(process.argv[2]) : DEFAULT_AMOUNT;
const currency = process.argv[3] || DEFAULT_CURRENCY;

// Coinbase Sandbox API credentials
const COINBASE_SANDBOX_API_KEY = 'Ajp6YwWIl6Xcq2KP3gyug33mgzx3UmQ77QuxiDLTBdLOa8XQVs+B+50j47rClYx/';
const COINBASE_SANDBOX_API_SECRET = 'IfB//R/x6Ydn5eo/nXF7gw==';

async function fundSandboxAccount() {
  console.log('🚀 Coinbase Sandbox Account Funding Script');
  console.log('==========================================');
  console.log(`Amount: $${amount.toLocaleString()} ${currency}`);
  console.log('');

  try {
    // Initialize Coinbase provider
    const coinbaseProvider = new CoinbaseCryptoProvider({
      apiKey: COINBASE_SANDBOX_API_KEY,
      apiSecret: COINBASE_SANDBOX_API_SECRET,
      isSandbox: true
    });

    console.log('📡 Connecting to Coinbase Sandbox API...');

    // Check if provider is healthy
    const isHealthy = await coinbaseProvider.isHealthy();
    if (!isHealthy) {
      throw new Error('Coinbase API is not responding');
    }
    console.log('✅ Coinbase API connection successful');

    // Get current balances before funding
    console.log('📊 Checking current balances...');
    const balancesBefore = await coinbaseProvider.getBalances();
    const usdBalance = balancesBefore[currency];
    console.log(`💰 Current ${currency} balance: $${(usdBalance?.total || 0).toFixed(2)}`);

    // Fund the account
    console.log(`💸 Funding account with $${amount.toLocaleString()} ${currency}...`);
    const fundingResult = await coinbaseProvider.fundSandboxAccount(currency, amount);

    console.log('✅ Funding successful!');
    console.log('📄 Transaction details:', fundingResult);

    // Wait a moment for the funding to process
    console.log('⏳ Waiting for transaction to process...');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Get balances after funding
    console.log('📊 Checking updated balances...');
    const balancesAfter = await coinbaseProvider.getBalances();
    const usdBalanceAfter = balancesAfter[currency];
    console.log(`💰 Updated ${currency} balance: $${(usdBalanceAfter?.total || 0).toFixed(2)}`);

    // Calculate the difference
    const difference = (usdBalanceAfter?.total || 0) - (usdBalance?.total || 0);
    console.log(`📈 Balance change: +$${difference.toFixed(2)} ${currency}`);

    console.log('');
    console.log('🎉 Sandbox account funding completed successfully!');
    console.log('💡 You can now use this account for paper trading on Coinbase Sandbox.');

  } catch (error) {
    console.error('❌ Funding failed:', error.message);

    if (error.response) {
      console.error('API Response:', error.response.data);
      console.error('Status Code:', error.response.status);
    }

    console.log('');
    console.log('🔍 Troubleshooting tips:');
    console.log('1. Verify your Coinbase Sandbox API credentials');
    console.log('2. Make sure you\'re using the sandbox environment');
    console.log('3. Check your internet connection');
    console.log('4. Verify the API key has the necessary permissions');

    process.exit(1);
  }
}

// Handle script execution
if (require.main === module) {
  fundSandboxAccount().catch(error => {
    console.error('💥 Script execution failed:', error);
    process.exit(1);
  });
}

module.exports = { fundSandboxAccount };
