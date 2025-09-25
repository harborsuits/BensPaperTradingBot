import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { get, post } from '@/lib/api';

export interface OptionPosition {
  id: string;
  symbol: string;
  underlying: string;
  optionType: 'call' | 'put';
  strike: number;
  expiration: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  metadata?: {
    strategy?: string;
    isOption: boolean;
  };
}

export interface OptionsMetrics {
  totalContracts: number;
  openPositions: number;
  totalPremiumCollected: number;
  totalPnL: number;
  winRate: number;
  capitalAllocated: number;
  maxCapital: number;
}

export interface OptionOrder {
  symbol: string;
  optionType: 'call' | 'put';
  strike: number;
  expiration: string;
  quantity: number;
  orderType?: 'market' | 'limit';
  limitPrice?: number;
  action: 'buy' | 'sell';
}

// Hook to fetch options positions
export function useOptionsPositions() {
  return useQuery<OptionPosition[]>({
    queryKey: ['options', 'positions'],
    queryFn: async () => {
      // Get all paper positions and filter for options
      const data = await get<any>('/api/paper/positions');
      const positions = Array.isArray(data) ? data : [];
      
      // Filter for options (symbol contains expiration pattern)
      const optionPositions = positions.filter(p => 
        p.metadata?.isOption || 
        /[A-Z]+\d{6}[CP]\d{8}/.test(p.symbol)
      );
      
      return optionPositions.map(p => ({
        id: p.id || p.symbol,
        symbol: p.symbol,
        underlying: p.metadata?.underlying || extractUnderlying(p.symbol),
        optionType: extractOptionType(p.symbol),
        strike: p.metadata?.strike || extractStrike(p.symbol),
        expiration: p.metadata?.expiration || extractExpiration(p.symbol),
        quantity: p.quantity,
        entryPrice: p.cost_basis || p.avg_price || 0,
        currentPrice: p.current_price || 0,
        pnl: p.unrealized_pl || 0,
        pnlPercent: p.unrealized_plpc || 0,
        metadata: p.metadata
      }));
    },
    refetchInterval: 30000
  });
}

// Hook to fetch options metrics
export function useOptionsMetrics() {
  const { data: positions } = useOptionsPositions();
  
  return useQuery<OptionsMetrics>({
    queryKey: ['options', 'metrics'],
    queryFn: async () => {
      // Calculate metrics from positions and trades
      const trades = await get<any>('/api/trades');
      const optionTrades = trades?.items?.filter(t => 
        t.metadata?.isOption || 
        /[A-Z]+\d{6}[CP]\d{8}/.test(t.symbol)
      ) || [];
      
      const totalPnL = optionTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
      const winningTrades = optionTrades.filter(t => (t.pnl || 0) > 0).length;
      const winRate = optionTrades.length > 0 ? winningTrades / optionTrades.length : 0;
      
      // Get capital info from AI policy
      const policy = await get<any>('/api/ai/policy');
      const optionsAllocation = policy?.options_allocation || 0.15;
      const totalCapital = policy?.paper_cap_max || 100000;
      
      return {
        totalContracts: positions?.reduce((sum, p) => sum + Math.abs(p.quantity), 0) || 0,
        openPositions: positions?.length || 0,
        totalPremiumCollected: optionTrades
          .filter(t => t.side === 'sell')
          .reduce((sum, t) => sum + (t.quantity * t.price * 100), 0),
        totalPnL,
        winRate,
        capitalAllocated: totalCapital * optionsAllocation,
        maxCapital: totalCapital * optionsAllocation
      };
    },
    enabled: !!positions,
    refetchInterval: 60000
  });
}

// Hook to place options orders
export function useOptionsOrder() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (order: OptionOrder) => {
      return post('/api/paper/options/order', order);
    },
    onSuccess: () => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: ['options'] });
      queryClient.invalidateQueries({ queryKey: ['paper', 'positions'] });
      queryClient.invalidateQueries({ queryKey: ['paper', 'orders'] });
    }
  });
}

// Hook for options chain analysis
export function useOptionsChain(symbol: string, expiration?: string) {
  return useQuery({
    queryKey: ['options', 'chain', symbol, expiration],
    queryFn: async () => {
      return post('/api/options/chain-analysis', {
        symbol,
        underlyingPrice: 0, // Will be filled by backend
        targetExpiration: expiration
      });
    },
    enabled: !!symbol && !!expiration
  });
}

// Helper functions
function extractUnderlying(optionSymbol: string): string {
  const match = optionSymbol.match(/^([A-Z]+)/);
  return match ? match[1] : optionSymbol;
}

function extractOptionType(optionSymbol: string): 'call' | 'put' {
  return optionSymbol.includes('C') ? 'call' : 'put';
}

function extractStrike(optionSymbol: string): number {
  const match = optionSymbol.match(/[CP](\d{8})$/);
  return match ? parseInt(match[1]) / 1000 : 0;
}

function extractExpiration(optionSymbol: string): string {
  const match = optionSymbol.match(/(\d{6})[CP]/);
  if (match) {
    const dateStr = match[1];
    const year = '20' + dateStr.slice(0, 2);
    const month = dateStr.slice(2, 4);
    const day = dateStr.slice(4, 6);
    return `${year}-${month}-${day}`;
  }
  return '';
}
