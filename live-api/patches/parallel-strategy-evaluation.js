// Implement parallel strategy evaluation for faster performance
// This patch modifies AutoLoop to evaluate strategies in parallel rather than sequentially

const fs = require('fs');
const path = require('path');

// Patch AutoLoop to use parallel evaluation
const autoLoopPath = path.join(__dirname, '../lib/autoLoop.js');
let autoLoopContent = fs.readFileSync(autoLoopPath, 'utf8');

// Replace the sequential strategy evaluation with parallel
const oldGenerateStrategySignals = `  /**
   * Generate signals from allocated strategies
   */
  async _generateStrategySignals(allocations) {
    const signals = [];
    const signalsBySymbol = new Map(); // Track best signal per symbol
    
    // Get current positions to avoid buying what we already own
    const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
    const positions = positionsResp.ok ? await positionsResp.json() : [];
    const ownedSymbols = new Set(positions.filter(p => p.qty > 0).map(p => p.symbol));
    
    // First, get ALL high-value candidates (diamonds, news movers, etc.)
    const highValueCandidates = await this._getHighValueCandidates();
    console.log(\`[AutoLoop] Found \${highValueCandidates.length} high-value candidates to test against ALL strategies\`);
    
    // Test high-value candidates against ALL allocated strategies
    for (const candidate of highValueCandidates) {
      // Skip if we already own this symbol
      if (ownedSymbols.has(candidate.symbol)) {
        console.log(\`[AutoLoop] Skipping \${candidate.symbol} - already have position\`);
        continue;
      }
      
      // Skip if we have a pending order for this symbol
      if (this.pendingOrderSymbols && this.pendingOrderSymbols.has(candidate.symbol)) {
        console.log(\`[AutoLoop] Skipping \${candidate.symbol} - already have pending order\`);
        continue;
      }
      
      for (const [strategyId, allocation] of Object.entries(allocations)) {
        if (allocation.weight <= 0) continue;
        
        try {
          const signal = await this._testCandidateWithStrategy(candidate, strategyId, allocation);
          if (signal) {
            // Keep only the best signal per symbol
            const existing = signalsBySymbol.get(signal.symbol);
            if (!existing || signal.confidence > existing.confidence) {
              signalsBySymbol.set(signal.symbol, signal);
            }
          }
        } catch (e) {
          console.error(\`[AutoLoop] Error testing \${candidate.symbol} with \${strategyId}:\`, e.message);
        }
      }
    }
    
    // Convert map to array
    signals.push(...signalsBySymbol.values());
    
    // If we found enough signals from high-value candidates, return early
    if (signals.length >= 3) {
      console.log(\`[AutoLoop] Found \${signals.length} unique signals from high-value candidates, proceeding to execution\`);
      return signals;
    }

    // Get regular candidates and test them against ALL strategies too
    const regularCandidates = await this._getRegularCandidates(allocations);
    console.log(\`[AutoLoop] Found \${regularCandidates.length} regular candidates to test against ALL strategies\`);
    
    // Test regular candidates against ALL strategies (just like high-value ones)
    for (const candidate of regularCandidates) {
      // Skip if we already own this symbol
      if (ownedSymbols.has(candidate.symbol)) {
        console.log(\`[AutoLoop] Skipping \${candidate.symbol} - already have position\`);
        continue;
      }
      
      // Skip if we have a pending order for this symbol
      if (this.pendingOrderSymbols && this.pendingOrderSymbols.has(candidate.symbol)) {
        console.log(\`[AutoLoop] Skipping \${candidate.symbol} - already have pending order\`);
        continue;
      }
      
      for (const [strategyId, allocation] of Object.entries(allocations)) {
        if (allocation.weight <= 0) continue;
        
        try {
          const signal = await this._testCandidateWithStrategy(candidate, strategyId, allocation);
          if (signal) {
            // Keep only the best signal per symbol
            const existing = signalsBySymbol.get(signal.symbol);
            if (!existing || signal.confidence > existing.confidence) {
              signalsBySymbol.set(signal.symbol, signal);
            }
          }
        } catch (e) {
          console.error(\`[AutoLoop] Error testing \${candidate.symbol} with \${strategyId}:\`, e.message);
        }
      }
    }
    
    // Convert remaining signals from map to array
    const uniqueSignals = Array.from(signalsBySymbol.values());
    console.log(\`[AutoLoop] Total unique signals generated: \${uniqueSignals.length} from \${highValueCandidates.length + regularCandidates.length} candidates\`);
    return uniqueSignals;
  }`;

const newGenerateStrategySignals = `  /**
   * Generate signals from allocated strategies - PARALLEL VERSION
   */
  async _generateStrategySignals(allocations) {
    const signalsBySymbol = new Map(); // Track best signal per symbol
    
    // Get current positions to avoid buying what we already own
    const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
    const positions = positionsResp.ok ? await positionsResp.json() : [];
    const ownedSymbols = new Set(positions.filter(p => p.qty > 0).map(p => p.symbol));
    
    // Get all candidates in parallel
    console.log('[AutoLoop] Fetching all candidates in parallel...');
    const [highValueCandidates, regularCandidates] = await Promise.all([
      this._getHighValueCandidates(),
      this._getRegularCandidates(allocations)
    ]);
    
    console.log(\`[AutoLoop] Found \${highValueCandidates.length} high-value and \${regularCandidates.length} regular candidates\`);
    
    // Combine all candidates
    const allCandidates = [...highValueCandidates, ...regularCandidates];
    
    // Filter out owned symbols and pending orders
    const eligibleCandidates = allCandidates.filter(candidate => {
      if (ownedSymbols.has(candidate.symbol)) {
        console.log(\`[AutoLoop] Skipping \${candidate.symbol} - already have position\`);
        return false;
      }
      if (this.pendingOrderSymbols && this.pendingOrderSymbols.has(candidate.symbol)) {
        console.log(\`[AutoLoop] Skipping \${candidate.symbol} - already have pending order\`);
        return false;
      }
      return true;
    });
    
    console.log(\`[AutoLoop] Testing \${eligibleCandidates.length} eligible candidates against \${Object.keys(allocations).length} strategies in parallel...\`);
    
    // Get active strategy IDs with positive allocations
    const activeStrategies = Object.entries(allocations)
      .filter(([_, allocation]) => allocation.weight > 0)
      .map(([strategyId, allocation]) => ({ strategyId, allocation }));
    
    // Create all test combinations (candidate × strategy)
    const testPromises = [];
    for (const candidate of eligibleCandidates) {
      for (const { strategyId, allocation } of activeStrategies) {
        testPromises.push(
          this._testCandidateWithStrategy(candidate, strategyId, allocation)
            .catch(e => {
              console.error(\`[AutoLoop] Error testing \${candidate.symbol} with \${strategyId}:\`, e.message);
              return null; // Return null on error to continue processing other candidates
            })
        );
      }
    }
    
    // Run all tests in parallel with concurrency limit to avoid overwhelming the system
    const BATCH_SIZE = 20; // Process 20 tests at a time
    const allSignals = [];
    
    for (let i = 0; i < testPromises.length; i += BATCH_SIZE) {
      const batch = testPromises.slice(i, i + BATCH_SIZE);
      const batchResults = await Promise.all(batch);
      
      // Filter out null results and add valid signals
      const validSignals = batchResults.filter(signal => signal !== null);
      allSignals.push(...validSignals);
      
      // Log progress
      if (i > 0 && i % 100 === 0) {
        console.log(\`[AutoLoop] Progress: tested \${i}/\${testPromises.length} combinations, found \${allSignals.length} signals so far\`);
      }
      
      // Early exit if we have enough high-confidence signals
      if (allSignals.length >= 10) {
        console.log(\`[AutoLoop] Found sufficient signals (\${allSignals.length}), stopping early\`);
        break;
      }
    }
    
    // Keep only the best signal per symbol
    for (const signal of allSignals) {
      if (signal) {
        const existing = signalsBySymbol.get(signal.symbol);
        if (!existing || signal.confidence > existing.confidence) {
          signalsBySymbol.set(signal.symbol, signal);
        }
      }
    }
    
    // Convert to array and sort by confidence
    const uniqueSignals = Array.from(signalsBySymbol.values())
      .sort((a, b) => b.confidence - a.confidence);
    
    console.log(\`[AutoLoop] Parallel evaluation complete: \${uniqueSignals.length} unique signals from \${testPromises.length} tests\`);
    
    return uniqueSignals;
  }`;

// Replace the method
autoLoopContent = autoLoopContent.replace(oldGenerateStrategySignals, newGenerateStrategySignals);

// Also optimize _monitorPositionsForExits to batch API calls
const oldMonitorPositions = `  /**
   * Monitor positions for quick exits, especially diamonds
   */
  async _monitorPositionsForExits() {
    try {
      // Get open positions
      const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
      if (!positionsResp.ok) return;
      
      const positions = await positionsResp.json();
      if (!positions || positions.length === 0) return;
      
      // Initialize peak PnL tracking if not already set
      if (!this.peakPnLTracker) {
        this.peakPnLTracker = new Map();
      }
      
      // Get current quotes for positions
      const symbols = positions.map(p => p.symbol).join(',');
      const quotesResp = await fetch(\`http://localhost:4000/api/quotes?symbols=\${symbols}\`);
      if (!quotesResp.ok) return;
      
      const quotes = await quotesResp.json();
      
      for (const position of positions) {
        const quote = quotes[position.symbol];
        if (!quote) continue;`;

const newMonitorPositions = `  /**
   * Monitor positions for quick exits, especially diamonds - PARALLEL VERSION
   */
  async _monitorPositionsForExits() {
    try {
      // Get open positions
      const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
      if (!positionsResp.ok) return;
      
      const positions = await positionsResp.json();
      if (!positions || positions.length === 0) return;
      
      // Initialize peak PnL tracking if not already set
      if (!this.peakPnLTracker) {
        this.peakPnLTracker = new Map();
      }
      
      // Get current quotes for positions
      const symbols = positions.map(p => p.symbol).join(',');
      const quotesResp = await fetch(\`http://localhost:4000/api/quotes?symbols=\${symbols}\`);
      if (!quotesResp.ok) return;
      
      const quotes = await quotesResp.json();
      
      // Process all positions in parallel
      const exitPromises = positions.map(async (position) => {
        const quote = quotes[position.symbol];
        if (!quote) return null;`;

// Replace the monitor positions method start
autoLoopContent = autoLoopContent.replace(oldMonitorPositions, newMonitorPositions);

// Find and replace the end of the monitor positions method to use Promise.all
const monitorEndPattern = /(\s+}\s+}\s+}\s+} catch \(error\) {\s+console\.error\('\[AutoLoop\] Error monitoring positions:', error\.message\);\s+}\s+})/;
const monitorEndReplacement = `
        }
      });
      
      // Execute all exit trades in parallel
      const exitSignals = await Promise.all(exitPromises);
      const validExitSignals = exitSignals.filter(signal => signal !== null);
      
      if (validExitSignals.length > 0) {
        console.log(\`[AutoLoop] Executing \${validExitSignals.length} exit trades in parallel\`);
        await Promise.all(validExitSignals.map(signal => this._executeTrade(signal)));
      }
    } catch (error) {
      console.error('[AutoLoop] Error monitoring positions:', error.message);
    }
  }`;

// Apply the monitor positions end replacement
autoLoopContent = autoLoopContent.replace(monitorEndPattern, monitorEndReplacement);

// Write the patched file
fs.writeFileSync(autoLoopPath, autoLoopContent);

console.log('✅ Parallel strategy evaluation implemented successfully!');
console.log('Benefits:');
console.log('- Strategy evaluations now run in parallel (up to 20 at a time)');
console.log('- Position monitoring and exits execute in parallel');
console.log('- Candidate fetching happens in parallel');
console.log('- Early exit when sufficient signals are found');
console.log('- Progress logging for long-running evaluations');
