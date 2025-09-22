#!/usr/bin/env node

const axios = require('axios');

async function checkTradierSync() {
  console.log('🔍 Checking Tradier Sync Status...\n');
  
  try {
    // Check portfolio summary
    const portfolioResp = await axios.get('http://localhost:4000/api/portfolio/summary');
    const portfolio = portfolioResp.data;
    
    console.log('📊 Portfolio Summary:');
    console.log(`   Broker: ${portfolio.broker}`);
    console.log(`   Mode: ${portfolio.mode}`);
    console.log(`   Cash: $${portfolio.cash?.toFixed(2) || 0}`);
    console.log(`   Equity: $${portfolio.equity?.toFixed(2) || 0}`);
    console.log(`   Positions: ${portfolio.positions?.length || 0}`);
    console.log(`   Day P&L: $${portfolio.day_pnl?.toFixed(2) || 0}`);
    console.log(`   As of: ${portfolio.asOf}\n`);
    
    // Check account details
    const accountResp = await axios.get('http://localhost:4000/api/paper/account');
    const account = accountResp.data;
    
    console.log('💰 Account Details:');
    console.log(`   Total Equity: $${account.balances?.total_equity?.toFixed(2) || 0}`);
    console.log(`   Total Cash: $${account.balances?.total_cash?.toFixed(2) || 0}`);
    console.log(`   Market Value: $${account.balances?.market_value?.toFixed(2) || 0}\n`);
    
    // Check positions
    const positionsResp = await axios.get('http://localhost:4000/api/paper/positions');
    const positions = positionsResp.data;
    
    console.log('📈 Positions:');
    if (positions.length === 0) {
      console.log('   No open positions');
    } else {
      positions.forEach(pos => {
        console.log(`   ${pos.symbol}: ${pos.qty} shares @ $${pos.avg_price?.toFixed(2)} (current: $${pos.current_price?.toFixed(2)})`);
      });
    }
    
    console.log('\n✅ Tradier sync is working!');
    
  } catch (error) {
    console.error('❌ Error checking Tradier sync:', error.message);
    if (error.response) {
      console.error('   Response:', error.response.data);
    }
  }
}

checkTradierSync();
