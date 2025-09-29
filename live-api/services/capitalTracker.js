/**
 * Real-Time Capital Commitment Tracker
 * Prevents overtrading by tracking all capital commitments across the system
 */

const EventEmitter = require('events');

class CapitalTracker extends EventEmitter {
  constructor(performanceRecorder, autoLoop) {
    super();
    this.performanceRecorder = performanceRecorder;
    this.autoLoop = autoLoop;
    
    // Capital tracking state
    this.state = {
      totalCapital: 100000, // Will be updated from account
      cashAvailable: 100000,
      openPositions: new Map(), // symbol -> { quantity, price, value }
      pendingOrders: new Map(), // orderId -> { symbol, quantity, price, estimatedValue }
      reservedCapital: 0,
      utilizationPct: 0,
      lastUpdated: new Date().toISOString()
    };
    
    // Limits from config
    this.limits = {
      maxUtilization: 0.95, // 95% max capital utilization
      maxPerTrade: 0.10,    // 10% per position
      reserveBuffer: 0.05,  // 5% cash reserve
      warningLevel: 0.80    // Warning at 80% utilization
    };
    
    // Update interval
    this.updateInterval = null;
    
    console.log('[CapitalTracker] Initialized');
  }
  
  /**
   * Start tracking capital
   */
  start() {
    // Update every 5 seconds
    this.updateInterval = setInterval(() => this.updateCapitalState(), 5000);
    this.updateCapitalState();
    
    // Listen for order events (AutoLoop doesn't extend EventEmitter, so we'll handle this differently)
    // Orders will be tracked through the broker events instead
    
    console.log('[CapitalTracker] Started tracking');
  }
  
  /**
   * Stop tracking
   */
  stop() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    console.log('[CapitalTracker] Stopped tracking');
  }
  
  /**
   * Update capital state from account data
   */
  async updateCapitalState() {
    try {
      // Get account data
      const account = await this.getAccountData();
      if (!account) return;
      
      // Update total capital and cash
      this.state.totalCapital = account.totalEquity || this.state.totalCapital;
      
      // Use actual cash from account if available
      if (account.totalCash !== undefined) {
        this.state.cashAvailable = account.totalCash;
      } else {
        // Calculate cash from equity minus positions
        let positionsValue = 0;
        if (account.positions) {
          for (const position of account.positions) {
            positionsValue += Math.abs(position.quantity * position.currentPrice);
          }
        }
        this.state.cashAvailable = this.state.totalCapital - positionsValue;
      }
      
      // Update open positions tracking
      this.state.openPositions.clear();
      if (account.positions) {
        for (const position of account.positions) {
          const value = Math.abs(position.quantity * position.currentPrice);
          this.state.openPositions.set(position.symbol, {
            quantity: position.quantity,
            price: position.currentPrice,
            value: value
          });
        }
      }
      
      // Subtract pending orders from available cash
      let pendingValue = 0;
      for (const [orderId, order] of this.state.pendingOrders) {
        pendingValue += order.estimatedValue;
      }
      
      // Reserve capital calculation
      this.state.reservedCapital = this.state.totalCapital * this.limits.reserveBuffer;
      
      // Calculate total committed capital (positions + pending orders)
      const committedCapital = this.getCommittedCapital();
      
      // Calculate utilization
      this.state.utilizationPct = committedCapital / this.state.totalCapital;
      this.state.lastUpdated = new Date().toISOString();
      
      // Emit warnings if needed
      if (this.state.utilizationPct >= this.limits.warningLevel) {
        this.emit('highUtilization', {
          utilization: this.state.utilizationPct,
          available: this.state.cashAvailable,
          threshold: this.limits.warningLevel
        });
      }
      
      // Log state periodically
      if (Math.random() < 0.1) { // Log 10% of the time
        console.log('[CapitalTracker] State:', {
          total: this.state.totalCapital.toFixed(2),
          committed: committedCapital.toFixed(2),
          available: this.state.cashAvailable.toFixed(2),
          utilization: (this.state.utilizationPct * 100).toFixed(1) + '%',
          positions: this.state.openPositions.size,
          pending: this.state.pendingOrders.size
        });
      }
      
    } catch (error) {
      console.error('[CapitalTracker] Update error:', error.message);
    }
  }
  
  /**
   * Check if a new trade can be placed
   */
  canPlaceTrade(symbol, quantity, price) {
    const estimatedValue = Math.abs(quantity * price);
    
    // Check per-trade limit
    if (estimatedValue > this.state.totalCapital * this.limits.maxPerTrade) {
      return {
        allowed: false,
        reason: `Trade size $${estimatedValue.toFixed(2)} exceeds max ${(this.limits.maxPerTrade * 100)}% of capital`
      };
    }
    
    // Check if it would exceed utilization limit
    const newUtilization = (this.getCommittedCapital() + estimatedValue) / this.state.totalCapital;
    if (newUtilization > this.limits.maxUtilization) {
      return {
        allowed: false,
        reason: `Would exceed ${(this.limits.maxUtilization * 100)}% utilization limit`
      };
    }
    
    // Check cash buffer
    const remainingCash = this.state.cashAvailable - estimatedValue;
    if (remainingCash < this.state.reservedCapital) {
      return {
        allowed: false,
        reason: `Would breach ${(this.limits.reserveBuffer * 100)}% cash reserve requirement`
      };
    }
    
    return { allowed: true };
  }
  
  /**
   * Handle order placed event
   */
  handleOrderPlaced(order) {
    const estimatedValue = Math.abs(order.quantity * order.price);
    this.state.pendingOrders.set(order.id, {
      symbol: order.symbol,
      quantity: order.quantity,
      price: order.price,
      estimatedValue: estimatedValue,
      timestamp: new Date().toISOString()
    });
    
    console.log(`[CapitalTracker] Order placed: ${order.symbol} $${estimatedValue.toFixed(2)}`);
    this.updateCapitalState();
  }
  
  /**
   * Handle order filled event
   */
  handleOrderFilled(order) {
    // Remove from pending
    this.state.pendingOrders.delete(order.id);
    
    // Update positions will happen in next updateCapitalState()
    console.log(`[CapitalTracker] Order filled: ${order.symbol}`);
    this.updateCapitalState();
  }
  
  /**
   * Handle order canceled event
   */
  handleOrderCanceled(orderId) {
    const order = this.state.pendingOrders.get(orderId);
    if (order) {
      console.log(`[CapitalTracker] Order canceled: ${order.symbol} $${order.estimatedValue.toFixed(2)}`);
      this.state.pendingOrders.delete(orderId);
      this.updateCapitalState();
    }
  }
  
  /**
   * Get total committed capital
   */
  getCommittedCapital() {
    let total = 0;
    
    // Open positions
    for (const [symbol, position] of this.state.openPositions) {
      total += position.value;
    }
    
    // Pending orders
    for (const [orderId, order] of this.state.pendingOrders) {
      total += order.estimatedValue;
    }
    
    return total;
  }
  
  /**
   * Get account data from paper trading account
   */
  async getAccountData() {
    try {
      // Fetch from paper account endpoint
      const axios = require('axios');
      const accountResp = await axios.get('http://localhost:4000/api/paper/account', {
        timeout: 5000  // 5 second timeout
      });
      
      const accountData = accountResp.data;
      if (accountData && accountData.balances) {
        const balances = accountData.balances;
        
        // Fetch positions separately
        let positions = [];
        try {
          const positionsResp = await axios.get('http://localhost:4000/api/paper/positions', {
            timeout: 5000
          });
          if (positionsResp.data && Array.isArray(positionsResp.data)) {
            positions = positionsResp.data.map(pos => ({
              symbol: pos.symbol,
              quantity: pos.qty || pos.quantity,
              currentPrice: pos.current_price || pos.price || 100,
              avgPrice: pos.avg_price || pos.price || 100
            }));
          }
        } catch (posErr) {
          console.log('[CapitalTracker] Could not fetch positions:', posErr.message);
        }
        
        return {
          totalEquity: balances.total_equity || 100000,
          totalCash: balances.total_cash || 100000,
          positions: positions
        };
      }
    } catch (error) {
      console.error('[CapitalTracker] Failed to get account data:', error.message);
    }
    
    // Fallback to default values if API fails
    return {
      totalEquity: 100000,
      totalCash: 100000,
      positions: []
    };
  }
  
  /**
   * Get current capital status
   */
  getStatus() {
    return {
      totalCapital: this.state.totalCapital,
      cashAvailable: this.state.cashAvailable,
      committedCapital: this.getCommittedCapital(),
      utilizationPct: this.state.utilizationPct,
      openPositions: this.state.openPositions.size,
      pendingOrders: this.state.pendingOrders.size,
      canTrade: this.state.utilizationPct < this.limits.maxUtilization,
      warnings: this.state.utilizationPct >= this.limits.warningLevel ? 
        [`High utilization: ${(this.state.utilizationPct * 100).toFixed(1)}%`] : [],
      lastUpdated: this.state.lastUpdated
    };
  }
  
  /**
   * Emergency capital release (cancel all pending orders)
   */
  async emergencyRelease() {
    console.warn('[CapitalTracker] EMERGENCY CAPITAL RELEASE TRIGGERED');
    
    const pendingCount = this.state.pendingOrders.size;
    const pendingValue = Array.from(this.state.pendingOrders.values())
      .reduce((sum, order) => sum + order.estimatedValue, 0);
    
    // Clear all pending orders
    this.state.pendingOrders.clear();
    
    // Request autoLoop to cancel orders if it has the method
    if (this.autoLoop && typeof this.autoLoop.cancelPendingOrders === 'function') {
      try {
        await this.autoLoop.cancelPendingOrders();
      } catch (error) {
        console.error('[CapitalTracker] Error canceling orders:', error.message);
      }
    }
    
    console.log(`[CapitalTracker] Released ${pendingCount} orders worth $${pendingValue.toFixed(2)}`);
    
    this.updateCapitalState();
    
    return {
      ordersReleased: pendingCount,
      capitalReleased: pendingValue
    };
  }
}

module.exports = CapitalTracker;
