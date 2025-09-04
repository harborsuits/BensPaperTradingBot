// Test script for paper orders endpoints on port 4001
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function testPaperOrders() {
  try {
    console.log('Testing /api/paper/orders/open endpoint on port 4001...');
    const openOrdersResponse = await fetch('http://localhost:4001/api/paper/orders/open');
    
    if (!openOrdersResponse.ok) {
      console.error(`Error: ${openOrdersResponse.status} ${openOrdersResponse.statusText}`);
      return;
    }
    
    const openOrders = await openOrdersResponse.json();
    console.log('Open orders:', JSON.stringify(openOrders, null, 2));
    
    console.log('\nTesting /api/paper/orders endpoint...');
    const allOrdersResponse = await fetch('http://localhost:4001/api/paper/orders');
    
    if (!allOrdersResponse.ok) {
      console.error(`Error: ${allOrdersResponse.status} ${allOrdersResponse.statusText}`);
      return;
    }
    
    const allOrders = await allOrdersResponse.json();
    console.log('All orders:', JSON.stringify(allOrders, null, 2));
    
  } catch (error) {
    console.error('Error testing paper orders:', error);
  }
}

testPaperOrders();


