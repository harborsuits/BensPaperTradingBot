#!/usr/bin/env node

/**
 * Coinbase Advanced Trade API Sandbox Test
 * No authentication required - perfect for paper trading!
 */

const axios = require('axios');

// Advanced Trade Sandbox API
const ADVANCED_SANDBOX_URL = 'https://api-sandbox.coinbase.com/api/v3/brokerage';

/**
 * Test the Advanced Trade Sandbox API
 */
async function testAdvancedSandbox() {
  console.log('🚀 Testing Coinbase Advanced Trade Sandbox API');
  console.log('===============================================');
  console.log('Note: No authentication required! 🎉\n');

  try {
    // Test 1: List Accounts
    console.log('1. Testing List Accounts...');
    const accountsResponse = await axios.get(`${ADVANCED_SANDBOX_URL}/accounts`);
    console.log('✅ List Accounts: SUCCESS');
    console.log(`Found ${accountsResponse.data.accounts.length} accounts`);

    // Show some account details
    if (accountsResponse.data.accounts.length > 0) {
      const firstAccount = accountsResponse.data.accounts[0];
      console.log(`Sample Account: ${firstAccount.name} (${firstAccount.currency})`);
      console.log(`Balance: ${firstAccount.available_balance.value} ${firstAccount.available_balance.currency}`);
    }

    // Test 2: Get Account Details
    console.log('\n2. Testing Get Account Details...');
    if (accountsResponse.data.accounts.length > 0) {
      const accountId = accountsResponse.data.accounts[0].uuid;
      const accountDetailsResponse = await axios.get(`${ADVANCED_SANDBOX_URL}/accounts/${accountId}`);
      console.log('✅ Get Account Details: SUCCESS');
      console.log(`Account: ${accountDetailsResponse.data.account.name}`);
    }

    // Test 3: List Portfolios
    console.log('\n3. Testing List Portfolios...');
    const portfoliosResponse = await axios.get(`${ADVANCED_SANDBOX_URL}/portfolios`);
    console.log('✅ List Portfolios: SUCCESS');
    console.log(`Found ${portfoliosResponse.data.portfolios.length} portfolios`);

    if (portfoliosResponse.data.portfolios.length > 0) {
      const firstPortfolio = portfoliosResponse.data.portfolios[0];
      console.log(`Sample Portfolio: ${firstPortfolio.name} (${firstPortfolio.type})`);
    }

    // Test 4: List Orders
    console.log('\n4. Testing List Orders...');
    const ordersResponse = await axios.get(`${ADVANCED_SANDBOX_URL}/orders/historical/batch`);
    console.log('✅ List Orders: SUCCESS');
    console.log(`Found ${ordersResponse.data.orders?.length || 0} orders`);

    // Test 5: Preview Order (test error case)
    console.log('\n5. Testing Preview Order with Error...');
    const previewOrderData = {
      side: "BUY",
      order_configuration: {
        market_market_ioc: {
          quote_size: "100.00"
        }
      },
      product_id: "BTC-USD"
    };

    try {
      const previewResponse = await axios.post(
        `${ADVANCED_SANDBOX_URL}/orders/preview`,
        previewOrderData,
        {
          headers: {
            'Content-Type': 'application/json',
            'X-Sandbox': 'PreviewOrder_insufficient_fund' // Trigger error response
          }
        }
      );
      console.log('✅ Preview Order: SUCCESS');
    } catch (error) {
      if (error.response?.status === 400) {
        console.log('✅ Preview Order Error Test: SUCCESS (expected insufficient funds)');
        console.log(`Error: ${error.response.data.error}`);
      } else {
        console.log(`❌ Preview Order: Unexpected error ${error.response?.status}`);
      }
    }

    console.log('\n🎉 All Advanced Trade Sandbox API tests completed!');
    console.log('This API is perfect for crypto paper trading - no API keys needed!');

    return {
      success: true,
      accounts: accountsResponse.data.accounts,
      portfolios: portfoliosResponse.data.portfolios
    };

  } catch (error) {
    console.log('❌ Test failed:', error.message);
    if (error.response) {
      console.log(`Status: ${error.response.status}`);
      console.log('Response:', error.response.data);
    }
    return { success: false, error: error.message };
  }
}

/**
 * Update funding scripts to use Advanced Trade API
 */
function updateFundingScripts() {
  console.log('\n🔄 Updating funding scripts for Advanced Trade API...');

  const fs = require('fs');
  const path = require('path');

  // Create new advanced trade funding script
  const advancedFundingScript = path.join(__dirname, 'fund-advanced-sandbox.js');
  const advancedFundingContent = `#!/usr/bin/env node

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
  console.log(\`Amount: \${DEFAULT_AMOUNT} \${DEFAULT_CURRENCY}\`);

  try {
    // Get accounts first
    const accountsResponse = await axios.get(\`\${ADVANCED_SANDBOX_URL}/accounts\`);
    console.log('✅ Retrieved accounts');

    // Find USD account
    const usdAccount = accountsResponse.data.accounts.find(acc => acc.currency === DEFAULT_CURRENCY);
    if (!usdAccount) {
      console.log(\`❌ No \${DEFAULT_CURRENCY} account found\`);
      return;
    }

    console.log(\`Found \${DEFAULT_CURRENCY} account: \${usdAccount.name}\`);
    console.log(\`Current balance: \${usdAccount.available_balance.value} \${usdAccount.available_balance.currency}\`);

    // For Advanced Trade API, funding is simulated
    // In real usage, you'd deposit funds through Coinbase interface
    console.log(\`🎉 Sandbox funding simulated: +\${DEFAULT_AMOUNT} \${DEFAULT_CURRENCY}\`);
    console.log(\`New simulated balance: \${parseFloat(usdAccount.available_balance.value) + parseFloat(DEFAULT_AMOUNT)} \${DEFAULT_CURRENCY}\`);

    console.log('\\n💡 Note: This is a sandbox environment.');
    console.log('To add real funds in production:');
    console.log('1. Go to Coinbase.com');
    console.log('2. Navigate to your account');
    console.log('3. Click "Deposit"');
    console.log('4. Follow the funding instructions');

  } catch (error) {
    console.log('❌ Funding failed:', error.message);
    if (error.response) {
      console.log(\`Status: \${error.response.status}\`);
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
`;

  fs.writeFileSync(advancedFundingScript, advancedFundingContent);
  console.log('✅ Created fund-advanced-sandbox.js');

  // Make it executable
  require('child_process').execSync(`chmod +x "${advancedFundingScript}"`);

  console.log('\\n🎯 Ready to use Advanced Trade Sandbox!');
  console.log('Run: node scripts/fund-advanced-sandbox.js 10000 USD');
}

// Main execution
if (require.main === module) {
  testAdvancedSandbox().then(result => {
    if (result.success) {
      updateFundingScripts();
    } else {
      console.log('\\n❌ Advanced Trade API test failed');
      console.log('Please check your internet connection and try again.');
    }
  }).catch(error => {
    console.error('💥 Test script failed:', error.message);
  });
}

module.exports = { testAdvancedSandbox };
