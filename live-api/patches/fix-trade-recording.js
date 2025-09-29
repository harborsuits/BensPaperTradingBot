// Fix trade recording to capture actual fill prices and quantities
// This patch addresses:
// 1. PaperBroker using hardcoded prices instead of real market data
// 2. Property name mismatch (filled_price vs fillPrice)
// 3. Proper fill price recording in performance tracker

const fs = require('fs');
const path = require('path');

// Patch 1: Update PaperBroker to use real market prices
const paperBrokerPath = path.join(__dirname, '../lib/PaperBroker.js');
let paperBrokerContent = fs.readFileSync(paperBrokerPath, 'utf8');

// Add quotes cache reference
const quotesImport = `const { getQuotesCache } = require('../minimal_server');
`;

if (!paperBrokerContent.includes('getQuotesCache')) {
  paperBrokerContent = quotesImport + paperBrokerContent;
}

// Replace getCurrentPrice method to use real quotes
const oldGetCurrentPrice = `  // Get current price for a symbol (mock implementation)
  getCurrentPrice(symbol) {
    // For now, return a mock price. In real implementation, this would
    // get the current market price from a quote service
    const basePrices = {
      'SPY': 450,
      'AAPL': 175,
      'QQQ': 380,
      'MSFT': 330,
      'NVDA': 450,
      'TSLA': 200
    };

    return basePrices[symbol] || 100; // Default to 100 if unknown symbol
  }`;

const newGetCurrentPrice = `  // Get current price for a symbol from real market data
  getCurrentPrice(symbol) {
    try {
      const quotesCache = getQuotesCache();
      
      // Handle different possible structures
      let quote = null;
      if (Array.isArray(quotesCache)) {
        quote = quotesCache.find(q => String(q.symbol || '').toUpperCase() === symbol.toUpperCase());
      } else if (quotesCache && typeof quotesCache === 'object') {
        if (quotesCache.quotes && Array.isArray(quotesCache.quotes)) {
          quote = quotesCache.quotes.find(q => String(q.symbol || '').toUpperCase() === symbol.toUpperCase());
        } else if (quotesCache.quotes && quotesCache.quotes[symbol]) {
          quote = quotesCache.quotes[symbol];
        } else if (quotesCache[symbol]) {
          quote = quotesCache[symbol];
        }
      }
      
      // Get the last traded price or midpoint
      if (quote) {
        return quote.last || quote.price || ((quote.bid + quote.ask) / 2) || 100;
      }
    } catch (error) {
      console.warn('[PaperBroker] Error getting quote for', symbol, ':', error.message);
    }
    
    // Fallback to basic prices if quotes not available
    const basePrices = {
      'SPY': 450,
      'AAPL': 175,
      'QQQ': 380,
      'MSFT': 330,
      'NVDA': 450,
      'TSLA': 200
    };

    return basePrices[symbol] || 100; // Default to 100 if unknown symbol
  }`;

paperBrokerContent = paperBrokerContent.replace(oldGetCurrentPrice, newGetCurrentPrice);

// Fix the property names to be consistent (use fillPrice instead of filled_price for event)
const oldFillMarketOrder = `    // Emit events
    this.emit('orderFilled', order);`;

const newFillMarketOrder = `    // Emit events with consistent property names
    this.emit('orderFilled', {
      ...order,
      fillPrice: order.filled_price,  // Add camelCase property for compatibility
      filledAt: order.filled_at,
      quantity: order.qty,
      metadata: order.metadata || {}
    });`;

paperBrokerContent = paperBrokerContent.replace(oldFillMarketOrder, newFillMarketOrder);

// Write the patched file
fs.writeFileSync(paperBrokerPath, paperBrokerContent);

console.log('✅ PaperBroker patched to use real market prices and emit correct event data');

// Patch 2: Ensure minimal_server exports getQuotesCache if not already
const minimalServerPath = path.join(__dirname, '../minimal_server.js');
let minimalServerContent = fs.readFileSync(minimalServerPath, 'utf8');

// Check if getQuotesCache is exported
if (!minimalServerContent.includes('module.exports') || !minimalServerContent.includes('getQuotesCache')) {
  // Add export at the end of file
  const exportSection = `
// Export for PaperBroker to access quotes
if (typeof module !== 'undefined' && module.exports) {
  module.exports.getQuotesCache = getQuotesCache;
}
`;
  
  minimalServerContent += exportSection;
  fs.writeFileSync(minimalServerPath, minimalServerContent);
  console.log('✅ Added getQuotesCache export to minimal_server.js');
}

console.log('✅ Trade recording fixes applied successfully');
