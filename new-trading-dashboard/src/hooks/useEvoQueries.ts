// Phase 3: EVO Queries & WS Integration
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';
import { useCallback } from 'react';

// Types
export type EvoStatus = 'staged' | 'activating' | 'active' | 'halting' | 'halted' | 'expired' | 'failed';

export type EvoAllocation = {
  id: string;
  status: EvoStatus;
  strategyRef: string;
  sessionId: string;
  allocation: number;
  ttlUntil: string;
  realizedPnl: number;
  symbolHint?: string[];
  createdAt: string;
};

export type EvoLedger = {
  ledgerVersion: string;
  rows: EvoAllocation[];
};

export type EvoPoolStatus = {
  capPct: number;
  utilizationPct: number;
  equity: number;
  poolPnl: number;
  activeCount: number;
  availableCapacity: number;
  riskLevel: 'low' | 'medium' | 'high';
  asOf: string;
};

export type PrecheckResult = {
  ok: boolean;
  failed: string[];
  metrics: {
    sharpe: number;
    maxDD: number;
    trades: number;
    win: number;
    slippageBps: number;
  };
};

// Query: EVO Ledger (allocations)
export const useEvoLedger = () => {
  return useQuery({
    queryKey: ['evo', 'ledger'],
    queryFn: () => fetch('/api/competition/ledger').then(r => r.json()) as Promise<EvoLedger>,
    refetchInterval: 30000,
    staleTime: 15000
  });
};

// Query: EVO Pool Status
export const useEvoPoolStatus = () => {
  return useQuery({
    queryKey: ['evo', 'pool'],
    queryFn: () => fetch('/api/competition/poolStatus').then(r => r.json()) as Promise<EvoPoolStatus>,
    refetchInterval: 30000,
    staleTime: 15000
  });
};

// Query: Precheck strategy for EVO promotion
export const useEvoPrecheck = (sessionId: string, strategyId: string) => {
  return useQuery({
    queryKey: ['evo', 'precheck', sessionId, strategyId],
    queryFn: () => fetch(`/api/competition/precheck/${sessionId}/${strategyId}`).then(r => r.json()) as Promise<PrecheckResult>,
    enabled: !!sessionId && !!strategyId,
    staleTime: 60000 // Precheck results are stable
  });
};

// Mutation: Stage strategy for EVO
export const useStageStrategy = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: {
      sessionId: string;
      strategyRef: string;
      allocation: number;
      pool: string;
      ttlDays: number;
    }) => fetch('/api/competition/stage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json()),

    // Optimistic update: Add staged allocation immediately
    onMutate: async (payload) => {
      await queryClient.cancelQueries(['evo', 'ledger']);
      const prev = queryClient.getQueryData<EvoLedger>(['evo', 'ledger']);

      const optimisticRow: EvoAllocation = {
        id: `temp_${Date.now()}`,
        status: 'staged',
        strategyRef: payload.strategyRef,
        sessionId: payload.sessionId,
        allocation: payload.allocation,
        ttlUntil: new Date(Date.now() + payload.ttlDays * 24 * 60 * 60 * 1000).toISOString(),
        realizedPnl: 0,
        createdAt: new Date().toISOString()
      };

      queryClient.setQueryData<EvoLedger>(['evo', 'ledger'], (old) => ({
        ledgerVersion: old?.ledgerVersion || '',
        rows: [...(old?.rows || []), optimisticRow]
      }));

      return { prev };
    },

    onError: (_e, _req, ctx) => {
      ctx?.prev && queryClient.setQueryData(['evo', 'ledger'], ctx.prev);
    },

    onSettled: () => {
      queryClient.invalidateQueries(['evo', 'ledger']);
    }
  });
};

// Mutation: Rebalance allocations
export const useRebalanceAllocations = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: {
      mode?: 'preview' | 'execute';
      ledgerVersion?: string;
    } = {}) => fetch('/api/competition/rebalance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json()),

    onSettled: () => {
      queryClient.invalidateQueries(['evo', 'ledger']);
      queryClient.invalidateQueries(['evo', 'pool']);
    }
  });
};

// WebSocket Event Handlers
export const useEvoWebSocketHandlers = () => {
  const queryClient = useQueryClient();

  // Allocation created - invalidate ledger
  useWebSocketMessage('evo_allocation_created', useCallback(() => {
    queryClient.invalidateQueries(['evo', 'ledger']);
  }, [queryClient]));

  // Rebalance complete - invalidate both ledger and pool
  useWebSocketMessage('evo_rebalance_complete', useCallback(() => {
    queryClient.invalidateQueries(['evo', 'ledger']);
    queryClient.invalidateQueries(['evo', 'pool']);
  }, [queryClient]));

  // Pool status changed - invalidate pool
  useWebSocketMessage('evo_pool_status', useCallback(() => {
    queryClient.invalidateQueries(['evo', 'pool']);
  }, [queryClient]));
};

// State machine utilities
export const canTransition = (from: EvoStatus, to: EvoStatus): boolean => {
  const validTransitions: Record<EvoStatus, EvoStatus[]> = {
    staged: ['activating', 'expired', 'failed'],
    activating: ['active', 'failed'],
    active: ['halting', 'expired', 'failed'],
    halting: ['halted', 'failed'],
    halted: ['expired'],
    expired: [],
    failed: []
  };

  return validTransitions[from]?.includes(to) ?? false;
};

export const getValidActions = (status: EvoStatus): string[] => {
  switch (status) {
    case 'staged':
      return ['cancel'];
    case 'active':
      return ['halt'];
    case 'halting':
      return [];
    case 'halted':
      return ['renew'];
    case 'expired':
    case 'failed':
      return ['remove'];
    default:
      return [];
  }
};
