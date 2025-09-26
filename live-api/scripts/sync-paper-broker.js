#!/usr/bin/env node

const axios = require('axios');

async function syncPaperBroker() {
  try {
    console.log('Syncing paper broker with Tradier...');
    
    // Get Tradier config
    const tradierConfigFile = require('../../config/credentials/tradier.json');
    const tradierConfig = tradierConfigFile.paper || tradierConfigFile;
    const baseUrl = tradierConfig.base_url || 'https://sandbox.tradier.com/v1';
    
    // Get profile
    const profileResp = await axios.get(`${baseUrl}/user/profile`, {
      headers: {
        'Authorization': `Bearer ${tradierConfig.api_key}`,
        'Accept': 'application/json'
      }
    });
    
    const profileData = profileResp.data;
    const accountNumber = profileData?.profile?.account?.account_number;
    console.log('Account:', accountNumber);
    
    // Get balances
    const balancesResp = await axios.get(`${baseUrl}/accounts/${accountNumber}/balances`, {
      headers: {
        'Authorization': `Bearer ${tradierConfig.api_key}`,
        'Accept': 'application/json'
      }
    });
    
    const balanceData = balancesResp.data;
    console.log('Balances:', balanceData.balances);
    
    // Get positions
    const positionsResp = await axios.get(`${baseUrl}/accounts/${accountNumber}/positions`, {
      headers: {
        'Authorization': `Bearer ${tradierConfig.api_key}`,
        'Accept': 'application/json'
      }
    });
    
    const positionsData = positionsResp.data;
    const positions = positionsData?.positions?.position || [];
    const positionArray = Array.isArray(positions) ? positions : [positions];
    console.log('Positions count:', positionArray.filter(p => p).length);
    
    // Update the paper broker data file
    const fs = require('fs');
    const path = require('path');
    
    // Format positions for Map reconstruction
    const positionsArray = [];
    positionArray.filter(p => p).forEach(pos => {
      positionsArray.push([pos.symbol, {
        quantity: Number(pos.quantity),
        avg_price: Number(pos.cost_basis) / Number(pos.quantity),
        total_cost: Number(pos.cost_basis)
      }]);
    });
    
    // Format data to match paperTradingEngine expectations
    const paperAccountData = {
      usd_balance: balanceData.balances.total_cash || 0,
      positions: positionsArray,
      orders: [],
      orderHistory: [],
      trades: [],
      lastOrderId: 1000
    };
    
    // Write to paper-account.json in the format expected by paperTradingEngine
    const dataDir = path.join(__dirname, '../data');
    fs.writeFileSync(path.join(dataDir, 'paper-account.json'), JSON.stringify(paperAccountData, null, 2));
    
    // Also save separate files for reference
    fs.writeFileSync(path.join(dataDir, 'paper-positions.json'), JSON.stringify(paperAccountData.positions, null, 2));
    fs.writeFileSync(path.join(dataDir, 'paper-orders.json'), JSON.stringify([], null, 2)); // Clear orders
    
    console.log('✅ Sync complete!');
    console.log('Cash:', paperBrokerData.account.cash);
    console.log('Equity:', paperBrokerData.account.equity);
    console.log('Positions:', Object.keys(paperBrokerData.positions).join(', '));
    
  } catch (error) {
    console.error('Sync error:', error);
  }
}

syncPaperBroker();
