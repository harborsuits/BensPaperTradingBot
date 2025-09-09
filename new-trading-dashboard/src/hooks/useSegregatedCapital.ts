import { useState, useEffect, useCallback } from 'react';
import segregatedCapitalService, {
  CapitalPool,
  CapitalAllocation,
  CapitalTransaction,
  CapitalLimits
} from '@/services/segregatedCapitalService';

interface UseSegregatedCapitalReturn {
  pools: CapitalPool[];
  allocations: CapitalAllocation[];
  transactions: CapitalTransaction[];
  limits: CapitalLimits;

  // Actions
  allocateCapital: (
    poolId: string,
    experimentId: string,
    amount: number,
    riskLevel: 'low' | 'medium' | 'high'
  ) => Promise<CapitalAllocation | null>;

  releaseCapital: (allocationId: string, finalPnl?: number) => Promise<void>;
  updatePnl: (allocationId: string, pnlChange: number) => void;
  transferCapital: (fromPoolId: string, toPoolId: string, amount: number, reason: string) => Promise<void>;
  updateLimits: (newLimits: Partial<CapitalLimits>) => void;

  // Analytics
  getPoolAnalytics: (poolId: string) => any;
  getActiveAllocations: (poolId?: string) => CapitalAllocation[];
  getPoolUtilization: (poolId: string) => number;
}

export const useSegregatedCapital = (): UseSegregatedCapitalReturn => {
  const [pools, setPools] = useState<CapitalPool[]>([]);
  const [allocations, setAllocations] = useState<CapitalAllocation[]>([]);
  const [transactions, setTransactions] = useState<CapitalTransaction[]>([]);
  const [limits, setLimits] = useState<CapitalLimits>(segregatedCapitalService.getLimits());

  // Load initial data
  useEffect(() => {
    refreshData();
  }, []);

  // Periodic updates
  useEffect(() => {
    const interval = setInterval(() => {
      refreshData();
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const refreshData = useCallback(() => {
    setPools(segregatedCapitalService.getPools());
    setAllocations(segregatedCapitalService.getAllocations());
    setTransactions(segregatedCapitalService.getTransactions());
    setLimits(segregatedCapitalService.getLimits());
  }, []);

  const allocateCapital = useCallback(async (
    poolId: string,
    experimentId: string,
    amount: number,
    riskLevel: 'low' | 'medium' | 'high' = 'low'
  ): Promise<CapitalAllocation | null> => {
    try {
      const allocation = segregatedCapitalService.allocateCapital(poolId, experimentId, amount, riskLevel);
      refreshData();
      return allocation;
    } catch (error) {
      console.error('Failed to allocate capital:', error);
      throw error;
    }
  }, [refreshData]);

  const releaseCapital = useCallback(async (allocationId: string, finalPnl: number = 0): Promise<void> => {
    try {
      segregatedCapitalService.releaseCapital(allocationId, finalPnl);
      refreshData();
    } catch (error) {
      console.error('Failed to release capital:', error);
      throw error;
    }
  }, [refreshData]);

  const updatePnl = useCallback((allocationId: string, pnlChange: number) => {
    segregatedCapitalService.updatePnl(allocationId, pnlChange);
    refreshData();
  }, [refreshData]);

  const transferCapital = useCallback(async (
    fromPoolId: string,
    toPoolId: string,
    amount: number,
    reason: string
  ): Promise<void> => {
    try {
      segregatedCapitalService.transferCapital(fromPoolId, toPoolId, amount, reason);
      refreshData();
    } catch (error) {
      console.error('Failed to transfer capital:', error);
      throw error;
    }
  }, [refreshData]);

  const updateLimits = useCallback((newLimits: Partial<CapitalLimits>) => {
    segregatedCapitalService.updateLimits(newLimits);
    setLimits(segregatedCapitalService.getLimits());
  }, []);

  const getPoolAnalytics = useCallback((poolId: string) => {
    return segregatedCapitalService.getPoolAnalytics(poolId);
  }, []);

  const getActiveAllocations = useCallback((poolId?: string) => {
    return segregatedCapitalService.getActiveAllocations(poolId);
  }, []);

  const getPoolUtilization = useCallback((poolId: string): number => {
    const pool = pools.find(p => p.id === poolId);
    if (!pool) return 0;
    return pool.allocatedCapital / pool.totalCapital;
  }, [pools]);

  return {
    pools,
    allocations,
    transactions,
    limits,

    // Actions
    allocateCapital,
    releaseCapital,
    updatePnl,
    transferCapital,
    updateLimits,

    // Analytics
    getPoolAnalytics,
    getActiveAllocations,
    getPoolUtilization
  };
};

export default useSegregatedCapital;
