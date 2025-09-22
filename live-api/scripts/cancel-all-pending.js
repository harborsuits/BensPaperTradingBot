#!/usr/bin/env node

const axios = require('axios');

async function cancelAllPendingOrders() {
  try {
    console.log('Fetching pending orders...');
    
    // Get all orders
    const ordersResp = await axios.get('http://localhost:4000/api/paper/orders?limit=100');
    const orders = ordersResp.data;
    const pendingOrders = orders.filter(o => o.status === 'pending');
    
    console.log(`Found ${pendingOrders.length} pending orders`);
    
    if (pendingOrders.length === 0) {
      console.log('No pending orders to cancel');
      return;
    }
    
    // Cancel each
    let cancelCount = 0;
    for (const order of pendingOrders.slice(0, 10)) { // Limit to first 10 for testing
      try {
        console.log(`Cancelling order ${order.id} for ${order.symbol}...`);
        const cancelResp = await axios.delete(`http://localhost:4000/api/paper/orders/${order.id}`);
        
        if (cancelResp.data.success) {
          cancelCount++;
          console.log(`✓ Cancelled order ${order.id}`);
        } else {
          console.log(`✗ Failed to cancel order ${order.id}`);
        }
      } catch (error) {
        console.error(`✗ Error cancelling order ${order.id}:`, error.message);
      }
      
      // Small delay
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    console.log(`\nCancelled ${cancelCount} orders`);
    
  } catch (error) {
    console.error('Error:', error.message);
  }
}

cancelAllPendingOrders();
