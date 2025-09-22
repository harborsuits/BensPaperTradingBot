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
    const paperBrokerData = {
      account: {
        cash: balanceData.balances.total_cash || 0,
        equity: balanceData.balances.total_equity || 0
      },
      positions: {},
      orderHistory: []
    };
    
    // Add positions
    positionArray.filter(p => p).forEach(pos => {
      paperBrokerData.positions[pos.symbol] = {
        symbol: pos.symbol,
        quantity: pos.quantity,
        avg_price: pos.cost_basis / pos.quantity,
        current_price: pos.last || 0,
        pnl: 0
      };
    });
    
    // Write to paper broker data files
    fs.writeFileSync('../data/paper-account.json', JSON.stringify(paperBrokerData.account, null, 2));
    fs.writeFileSync('../data/paper-positions.json', JSON.stringify(paperBrokerData.positions, null, 2));
    fs.writeFileSync('../data/paper-orders.json', JSON.stringify([], null, 2)); // Clear orders
    
    console.log('✅ Sync complete!');
    console.log('Cash:', paperBrokerData.account.cash);
    console.log('Equity:', paperBrokerData.account.equity);
    console.log('Positions:', Object.keys(paperBrokerData.positions).join(', '));
    
  } catch (error) {
    console.error('Sync error:', error);
  }
}

syncPaperBroker();
