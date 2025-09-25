// Quick fix for phantom trades - add this to RSI strategy

// Before line 99 in RSIStrategy.js, add:
if (currentQty <= 0) {
  // Reset state if we think we should sell but have no position
  if (this.lastSignal === 'BUY') {
    console.log(`[RSI] Resetting state - no ${this.config.symbol} position found`);
    this.lastSignal = null;
  }
  return { signal: null, data: { message: 'No position to sell' } };
}

// This prevents trying to sell positions we don't have!
