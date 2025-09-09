export interface CapitalPool {
  id: string;
  name: string;
  totalCapital: number;
  availableCapital: number;
  allocatedCapital: number;
  lockedCapital: number;
  riskLevel: 'low' | 'medium' | 'high';
  purpose: 'research' | 'competition' | 'validation';
  maxDrawdown: number;
  currentDrawdown: number;
  lastUpdated: Date;
}

export interface CapitalAllocation {
  id: string;
  poolId: string;
  experimentId: string;
  amount: number;
  allocatedAt: Date;
  releasedAt?: Date;
  status: 'active' | 'released' | 'locked';
  pnl: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export interface CapitalTransaction {
  id: string;
  type: 'allocation' | 'release' | 'pnl_update' | 'transfer';
  poolId: string;
  experimentId?: string;
  amount: number;
  timestamp: Date;
  description: string;
  metadata?: Record<string, any>;
}

export interface CapitalLimits {
  maxPerExperiment: Record<string, number>; // risk level -> max amount
  maxConcurrentExperiments: number;
  maxTotalDrawdown: number;
  autoRebalance: boolean;
  emergencyStopLoss: number;
}

class SegregatedCapitalService {
  private pools: Map<string, CapitalPool> = new Map();
  private allocations: Map<string, CapitalAllocation> = new Map();
  private transactions: CapitalTransaction[] = [];
  private limits: CapitalLimits;

  constructor() {
    this.limits = {
      maxPerExperiment: {
        low: 1000,
        medium: 2500,
        high: 5000
      },
      maxConcurrentExperiments: 5,
      maxTotalDrawdown: 0.15, // 15%
      autoRebalance: true,
      emergencyStopLoss: 0.20 // 20%
    };

    this.initializePools();
  }

  private initializePools() {
    // Research Pool - Low risk, high frequency experiments
    this.createPool({
      id: 'research_pool',
      name: 'Research Pool',
      totalCapital: 10000,
      availableCapital: 7500,
      allocatedCapital: 2500,
      lockedCapital: 0,
      riskLevel: 'low',
      purpose: 'research',
      maxDrawdown: 0.10, // 10%
      currentDrawdown: 0.02,
      lastUpdated: new Date()
    });

    // Competition Pool - Main trading capital
    this.createPool({
      id: 'competition_pool',
      name: 'Competition Pool',
      totalCapital: 50000,
      availableCapital: 35000,
      allocatedCapital: 15000,
      lockedCapital: 0,
      riskLevel: 'medium',
      purpose: 'competition',
      maxDrawdown: 0.20, // 20%
      currentDrawdown: 0.05,
      lastUpdated: new Date()
    });

    // Validation Pool - For testing promoted strategies
    this.createPool({
      id: 'validation_pool',
      name: 'Validation Pool',
      totalCapital: 15000,
      availableCapital: 12000,
      allocatedCapital: 3000,
      lockedCapital: 0,
      riskLevel: 'medium',
      purpose: 'validation',
      maxDrawdown: 0.15, // 15%
      currentDrawdown: 0.01,
      lastUpdated: new Date()
    });
  }

  createPool(pool: CapitalPool) {
    this.pools.set(pool.id, pool);
    this.recordTransaction({
      id: `init_${pool.id}_${Date.now()}`,
      type: 'allocation',
      poolId: pool.id,
      amount: pool.totalCapital,
      timestamp: new Date(),
      description: `Initialize ${pool.name} with ${pool.totalCapital}`,
      metadata: { initialization: true }
    });
  }

  allocateCapital(
    poolId: string,
    experimentId: string,
    amount: number,
    riskLevel: 'low' | 'medium' | 'high' = 'low'
  ): CapitalAllocation | null {
    const pool = this.pools.get(poolId);
    if (!pool) {
      throw new Error(`Pool ${poolId} not found`);
    }

    // Check limits
    const maxAmount = this.limits.maxPerExperiment[riskLevel];
    if (amount > maxAmount) {
      throw new Error(`Amount ${amount} exceeds maximum ${maxAmount} for risk level ${riskLevel}`);
    }

    // Check available capital
    if (amount > pool.availableCapital) {
      throw new Error(`Insufficient capital in pool ${poolId}. Available: ${pool.availableCapital}, Requested: ${amount}`);
    }

    // Check concurrent experiments limit
    const activeAllocations = Array.from(this.allocations.values())
      .filter(a => a.poolId === poolId && a.status === 'active');
    if (activeAllocations.length >= this.limits.maxConcurrentExperiments) {
      throw new Error(`Maximum concurrent experiments (${this.limits.maxConcurrentExperiments}) reached for pool ${poolId}`);
    }

    // Create allocation
    const allocation: CapitalAllocation = {
      id: `alloc_${poolId}_${experimentId}_${Date.now()}`,
      poolId,
      experimentId,
      amount,
      allocatedAt: new Date(),
      status: 'active',
      pnl: 0,
      riskLevel
    };

    this.allocations.set(allocation.id, allocation);

    // Update pool
    pool.allocatedCapital += amount;
    pool.availableCapital -= amount;
    pool.lastUpdated = new Date();

    // Record transaction
    this.recordTransaction({
      id: allocation.id,
      type: 'allocation',
      poolId,
      experimentId,
      amount,
      timestamp: new Date(),
      description: `Allocated ${amount} to experiment ${experimentId}`,
      metadata: { riskLevel, allocationId: allocation.id }
    });

    return allocation;
  }

  releaseCapital(allocationId: string, finalPnl: number = 0) {
    const allocation = this.allocations.get(allocationId);
    if (!allocation) {
      throw new Error(`Allocation ${allocationId} not found`);
    }

    if (allocation.status !== 'active') {
      throw new Error(`Allocation ${allocationId} is not active`);
    }

    const pool = this.pools.get(allocation.poolId);
    if (!pool) {
      throw new Error(`Pool ${allocation.poolId} not found`);
    }

    // Update allocation
    allocation.status = 'released';
    allocation.releasedAt = new Date();
    allocation.pnl = finalPnl;

    // Update pool
    pool.allocatedCapital -= allocation.amount;
    pool.availableCapital += allocation.amount + finalPnl; // Return capital + pnl
    pool.lastUpdated = new Date();

    // Update drawdown
    if (finalPnl < 0) {
      pool.currentDrawdown = Math.abs(finalPnl) / pool.totalCapital;
    }

    // Record transaction
    this.recordTransaction({
      id: `release_${allocationId}_${Date.now()}`,
      type: 'release',
      poolId: allocation.poolId,
      experimentId: allocation.experimentId,
      amount: allocation.amount + finalPnl,
      timestamp: new Date(),
      description: `Released ${allocation.amount} from experiment ${allocation.experimentId} with P&L ${finalPnl}`,
      metadata: { allocationId, finalPnl }
    });

    return allocation;
  }

  updatePnl(allocationId: string, pnlChange: number) {
    const allocation = this.allocations.get(allocationId);
    if (!allocation) {
      return;
    }

    allocation.pnl += pnlChange;

    const pool = this.pools.get(allocation.poolId);
    if (pool) {
      pool.lastUpdated = new Date();

      // Check for emergency stop loss
      const totalPnl = Array.from(this.allocations.values())
        .filter(a => a.poolId === pool.id && a.status === 'active')
        .reduce((sum, a) => sum + a.pnl, 0);

      if (Math.abs(totalPnl) / pool.totalCapital > this.limits.emergencyStopLoss) {
        console.warn(`Emergency stop loss triggered for pool ${pool.id}`);
        this.emergencyStop(pool.id);
      }
    }

    // Record transaction
    this.recordTransaction({
      id: `pnl_${allocationId}_${Date.now()}`,
      type: 'pnl_update',
      poolId: allocation.poolId,
      experimentId: allocation.experimentId,
      amount: pnlChange,
      timestamp: new Date(),
      description: `P&L update for experiment ${allocation.experimentId}: ${pnlChange}`,
      metadata: { allocationId, pnlChange }
    });
  }

  private emergencyStop(poolId: string) {
    const activeAllocations = Array.from(this.allocations.values())
      .filter(a => a.poolId === poolId && a.status === 'active');

    for (const allocation of activeAllocations) {
      try {
        this.releaseCapital(allocation.id, allocation.pnl);
      } catch (error) {
        console.error(`Failed to emergency release allocation ${allocation.id}:`, error);
      }
    }
  }

  transferCapital(fromPoolId: string, toPoolId: string, amount: number, reason: string) {
    const fromPool = this.pools.get(fromPoolId);
    const toPool = this.pools.get(toPoolId);

    if (!fromPool || !toPool) {
      throw new Error('Invalid pool IDs');
    }

    if (fromPool.availableCapital < amount) {
      throw new Error(`Insufficient available capital in ${fromPoolId}`);
    }

    // Update pools
    fromPool.availableCapital -= amount;
    toPool.availableCapital += amount;
    fromPool.lastUpdated = new Date();
    toPool.lastUpdated = new Date();

    // Record transaction
    this.recordTransaction({
      id: `transfer_${fromPoolId}_${toPoolId}_${Date.now()}`,
      type: 'transfer',
      poolId: fromPoolId,
      amount: -amount,
      timestamp: new Date(),
      description: `Transfer ${amount} from ${fromPoolId} to ${toPoolId}: ${reason}`,
      metadata: { toPoolId, reason }
    });

    this.recordTransaction({
      id: `transfer_${toPoolId}_${fromPoolId}_${Date.now()}`,
      type: 'transfer',
      poolId: toPoolId,
      amount: amount,
      timestamp: new Date(),
      description: `Transfer ${amount} from ${fromPoolId} to ${toPoolId}: ${reason}`,
      metadata: { fromPoolId, reason }
    });
  }

  private recordTransaction(transaction: CapitalTransaction) {
    this.transactions.push(transaction);

    // Keep only last 1000 transactions
    if (this.transactions.length > 1000) {
      this.transactions = this.transactions.slice(-1000);
    }
  }

  // Public getters
  getPools(): CapitalPool[] {
    return Array.from(this.pools.values());
  }

  getPool(poolId: string): CapitalPool | undefined {
    return this.pools.get(poolId);
  }

  getAllocations(poolId?: string): CapitalAllocation[] {
    const allocations = Array.from(this.allocations.values());
    return poolId ? allocations.filter(a => a.poolId === poolId) : allocations;
  }

  getActiveAllocations(poolId?: string): CapitalAllocation[] {
    const allocations = this.getAllocations(poolId);
    return allocations.filter(a => a.status === 'active');
  }

  getTransactions(poolId?: string, limit: number = 50): CapitalTransaction[] {
    let transactions = this.transactions;
    if (poolId) {
      transactions = transactions.filter(t => t.poolId === poolId);
    }
    return transactions.slice(-limit);
  }

  getLimits(): CapitalLimits {
    return { ...this.limits };
  }

  updateLimits(newLimits: Partial<CapitalLimits>) {
    this.limits = { ...this.limits, ...newLimits };
  }

  // Analytics
  getPoolAnalytics(poolId: string) {
    const pool = this.pools.get(poolId);
    if (!pool) return null;

    const allocations = this.getAllocations(poolId);
    const activeAllocations = allocations.filter(a => a.status === 'active');
    const completedAllocations = allocations.filter(a => a.status === 'released');

    const totalPnl = completedAllocations.reduce((sum, a) => sum + a.pnl, 0);
    const avgPnl = completedAllocations.length > 0 ? totalPnl / completedAllocations.length : 0;
    const winRate = completedAllocations.length > 0
      ? completedAllocations.filter(a => a.pnl > 0).length / completedAllocations.length
      : 0;

    return {
      poolId,
      totalPnl,
      avgPnl,
      winRate,
      activeExperiments: activeAllocations.length,
      completedExperiments: completedAllocations.length,
      utilizationRate: pool.allocatedCapital / pool.totalCapital,
      drawdownPercent: pool.currentDrawdown,
      riskAdjustedReturn: avgPnl / (pool.currentDrawdown || 0.01)
    };
  }

  // Cleanup
  destroy() {
    this.pools.clear();
    this.allocations.clear();
    this.transactions = [];
  }
}

// Export singleton instance
export const segregatedCapitalService = new SegregatedCapitalService();
export default segregatedCapitalService;
