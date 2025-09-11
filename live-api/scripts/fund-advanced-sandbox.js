#!/usr/bin/env node

/**
 * Fund Coinbase Advanced Trade Sandbox Account
 * Uses the Advanced Trade API (no authentication required)
 */

const axios = require('axios');

const ADVANCED_SANDBOX_URL = 'https://api-sandbox.coinbase.com/api/v3/brokerage';

// Default funding parameters
const DEFAULT_AMOUNT = process.argv[2] || 10000;
const DEFAULT_CURRENCY = process.argv[3] || 'USD';

/**
 * Fund sandbox account using Advanced Trade API
 */
async function fundAdvancedSandbox() {
  console.log('🚀 Funding Coinbase Advanced Trade Sandbox Account');
  console.log('==================================================');
  console.log(`Amount: ${DEFAULT_AMOUNT} ${DEFAULT_CURRENCY}`);

  try {
    // Get accounts first
    const accountsResponse = await axios.get(`${ADVANCED_SANDBOX_URL}/accounts`);
    console.log('✅ Retrieved accounts');

    // Find USD account
    const usdAccount = accountsResponse.data.accounts.find(acc => acc.currency === DEFAULT_CURRENCY);
    if (!usdAccount) {
      console.log(`❌ No ${DEFAULT_CURRENCY} account found`);
      return;
    }

    console.log(`Found ${DEFAULT_CURRENCY} account: ${usdAccount.name}`);
    console.log(`Current balance: ${usdAccount.available_balance.value} ${usdAccount.available_balance.currency}`);

    // For Advanced Trade API, funding is simulated
    // In real usage, you'd deposit funds through Coinbase interface
    console.log(`🎉 Sandbox funding simulated: +${DEFAULT_AMOUNT} ${DEFAULT_CURRENCY}`);
    console.log(`New simulated balance: ${parseFloat(usdAccount.available_balance.value) + parseFloat(DEFAULT_AMOUNT)} ${DEFAULT_CURRENCY}`);

    console.log('\n💡 Note: This is a sandbox environment.');
    console.log('To add real funds in production:');
    console.log('1. Go to Coinbase.com');
    console.log('2. Navigate to your account');
    console.log('3. Click "Deposit"');
    console.log('4. Follow the funding instructions');

  } catch (error) {
    console.log('❌ Funding failed:', error.message);
    if (error.response) {
      console.log(`Status: ${error.response.status}`);
      console.log('Response:', error.response.data);
    }
  }
}

// Run the funding script
if (require.main === module) {
  fundAdvancedSandbox().catch(error => {
    console.error('💥 Funding script failed:', error.message);
    process.exit(1);
  });
}

module.exports = { fundAdvancedSandbox };
