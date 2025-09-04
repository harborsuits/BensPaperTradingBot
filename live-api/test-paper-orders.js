// Test script for paper orders endpoints
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function testPaperOrders() {
  try {
    console.log('Testing /api/paper/orders/open endpoint...');
    const openOrdersResponse = await fetch('http://localhost:4000/api/paper/orders/open');
    
    if (!openOrdersResponse.ok) {
      console.error(`Error: ${openOrdersResponse.status} ${openOrdersResponse.statusText}`);
      return;
    }
    
    const openOrders = await openOrdersResponse.json();
    console.log('Open orders:', JSON.stringify(openOrders, null, 2));
    
    console.log('\nTesting /api/paper/orders endpoint...');
    const allOrdersResponse = await fetch('http://localhost:4000/api/paper/orders');
    
    if (!allOrdersResponse.ok) {
      console.error(`Error: ${allOrdersResponse.status} ${allOrdersResponse.statusText}`);
      return;
    }
    
    const allOrders = await allOrdersResponse.json();
    console.log('All orders:', JSON.stringify(allOrders, null, 2));
    
    // Create a test order
    console.log('\nCreating a test paper order...');
    const createOrderResponse = await fetch('http://localhost:4000/api/paper/orders', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: 'AAPL',
        side: 'buy',
        qty: 10,
        type: 'market'
      })
    });
    
    if (!createOrderResponse.ok) {
      console.error(`Error: ${createOrderResponse.status} ${createOrderResponse.statusText}`);
      return;
    }
    
    const createdOrder = await createOrderResponse.json();
    console.log('Created order:', JSON.stringify(createdOrder, null, 2));
    
    // Check open orders again
    console.log('\nChecking open orders after creation...');
    const openOrdersAfterResponse = await fetch('http://localhost:4000/api/paper/orders/open');
    const openOrdersAfter = await openOrdersAfterResponse.json();
    console.log('Open orders after creation:', JSON.stringify(openOrdersAfter, null, 2));
    
  } catch (error) {
    console.error('Error testing paper orders:', error);
  }
}

testPaperOrders();