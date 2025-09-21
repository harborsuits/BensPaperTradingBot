const { TradierQuoteProvider } = require('./src/providers/quotes/tradier');

async function testRealQuotes() {
  const provider = new TradierQuoteProvider(process.env.TRADIER_TOKEN);
  
  console.log('Testing real Tradier quotes...');
  try {
    const quotes = await provider.getQuotes(['NVDA', 'TSLA']);
    console.log('Real quotes result:', JSON.stringify(quotes, null, 2));
  } catch (error) {
    console.log('Error getting real quotes:', error.message);
    console.log('This is expected if TRADIER_TOKEN is not set to a valid token');
  }
}

testRealQuotes();
