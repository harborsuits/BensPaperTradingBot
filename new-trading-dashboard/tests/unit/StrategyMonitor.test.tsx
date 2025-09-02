import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import StrategyMonitor from '../../src/components/strategy/StrategyMonitor';
import { strategyApi } from '../../src/services/strategyApi';
import { toast } from 'react-hot-toast';

// Define types for TypeScript
interface StrategyPerformance {
  daily: number;
  weekly: number;
  monthly: number;
  overall: number;
}

interface SignalData {
  direction: 'long' | 'short' | 'neutral';
  strength: number;
  timestamp: string;
}

interface Strategy {
  id: string;
  name: string;
  description: string;
  category: string;
  enabled: boolean;
  performance: StrategyPerformance;
  marketSuitability: number;
  parameters: Record<string, any>;
  status: string;
  lastSignal: SignalData;
}

// Mock dependencies
jest.mock('../../src/services/strategyApi', () => ({
  strategyApi: {
    getActiveStrategies: jest.fn(),
    toggleStrategy: jest.fn()
  }
}));
jest.mock('react-hot-toast');

// Create a wrapper component with the required providers
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('StrategyMonitor Component', () => {
  const mockStrategies = [
    {
      id: 'strategy-1',
      name: 'VWAP Momentum',
      description: 'Follows momentum using VWAP as reference',
      category: 'momentum',
      enabled: true,
      performance: {
        daily: 1.25,
        weekly: -0.42,
        monthly: 3.17,
        overall: 12.53
      },
      marketSuitability: 0.78,
      parameters: { lookback: 20, threshold: 0.02 },
      status: 'active',
      lastSignal: {
        direction: 'long',
        strength: 0.85,
        timestamp: new Date(Date.now() - 1200000).toISOString()
      }
    },
    {
      id: 'strategy-2',
      name: 'Mean Reversion',
      description: 'Trades mean reversion on oversold conditions',
      category: 'reversion',
      enabled: false,
      performance: {
        daily: -0.35,
        weekly: 1.22,
        monthly: 2.45,
        overall: 8.73
      },
      marketSuitability: 0.45,
      parameters: { bands: 2.5, period: 14 },
      status: 'idle',
      lastSignal: {
        direction: 'neutral',
        strength: 0.2,
        timestamp: new Date(Date.now() - 3600000).toISOString()
      }
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock strategy API
    (strategyApi.getActiveStrategies as jest.Mock).mockResolvedValue({
      success: true,
      data: mockStrategies
    });
    
    (strategyApi.toggleStrategy as jest.Mock).mockImplementation((id) => {
      const strategy = mockStrategies.find(s => s.id === id);
      return Promise.resolve({
        success: true,
        data: {
          success: true,
          strategy: {
            ...strategy,
            enabled: !strategy?.enabled
          }
        }
      });
    });
  });

  test('renders strategies correctly', async () => {
    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('VWAP Momentum')).toBeInTheDocument();
    });
    
    // Check if both strategies are displayed
    expect(screen.getByText('VWAP Momentum')).toBeInTheDocument();
    expect(screen.getByText('Mean Reversion')).toBeInTheDocument();
    
    // Check category badges
    expect(screen.getByText('momentum')).toBeInTheDocument();
    expect(screen.getByText('reversion')).toBeInTheDocument();
    
    // Check status badges
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(screen.getByText('Idle')).toBeInTheDocument();
  });

  test('shows loading state initially', () => {
    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Check loading skeleton is displayed
    expect(screen.getByText('Active Strategies')).toBeInTheDocument();
    expect(document.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
  });

  test('expands strategy details when clicked', async () => {
    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('VWAP Momentum')).toBeInTheDocument();
    });
    
    // Performance details should not be visible initially
    expect(screen.queryByText('Daily')).not.toBeInTheDocument();
    
    // Find and click the settings button for the first strategy
    const settingsButtons = document.querySelectorAll('button.text-muted-foreground');
    fireEvent.click(settingsButtons[0]);
    
    // Now performance details should be visible
    expect(screen.getByText('Daily')).toBeInTheDocument();
    expect(screen.getByText('+1.25%')).toBeInTheDocument();
    expect(screen.getByText('Weekly')).toBeInTheDocument();
    expect(screen.getByText('-0.42%')).toBeInTheDocument();
    
    // Market suitability section should be visible
    expect(screen.getByText('Market Suitability Score')).toBeInTheDocument();
  });

  test('toggles strategy enabled status', async () => {
    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Mean Reversion')).toBeInTheDocument();
    });
    
    // Find and click the switch for the second strategy (Mean Reversion)
    const switches = document.querySelectorAll('input[type="checkbox"]');
    fireEvent.click(switches[1]);
    
    // Verify API was called with correct strategy ID
    expect(strategyApi.toggleStrategy).toHaveBeenCalledWith('strategy-2');
    
    // Check toast was shown
    await waitFor(() => {
      expect(toast.success).toHaveBeenCalledWith(
        expect.stringContaining('Strategy enabled successfully')
      );
    });
  });

  test('handles error state correctly', async () => {
    // Mock API error
    (strategyApi.getActiveStrategies as jest.Mock).mockResolvedValue({
      success: false,
      error: 'Failed to fetch strategy data'
    });

    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Wait for error state
    await waitFor(() => {
      expect(screen.getByText('Could not load strategy data')).toBeInTheDocument();
    });
    
    // Check retry button exists
    const retryButton = screen.getByText('Retry');
    expect(retryButton).toBeInTheDocument();
    
    // Reset mock
    (strategyApi.getActiveStrategies as jest.Mock).mockResolvedValue({
      success: true,
      data: mockStrategies
    });
    
    // Click retry
    fireEvent.click(retryButton);
    
    // Verify API was called again
    expect(strategyApi.getActiveStrategies).toHaveBeenCalledTimes(2);
  });

  test('displays empty state when no strategies', async () => {
    // Mock empty strategies
    (strategyApi.getActiveStrategies as jest.Mock).mockResolvedValue({
      success: true,
      data: []
    });
    
    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Wait for empty state
    await waitFor(() => {
      expect(screen.getByText('No active strategies')).toBeInTheDocument();
    });
  });

  test('refresh button refetches data', async () => {
    render(<StrategyMonitor />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('VWAP Momentum')).toBeInTheDocument();
    });
    
    // Find and click refresh button
    const refreshButton = document.querySelector('button[aria-label="Refresh"]') || 
                         document.querySelector('button.absolute.top-2.right-2');
    fireEvent.click(refreshButton as HTMLElement);
    
    // Verify toast and refetch
    expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('Refreshing strategy data'));
    expect(strategyApi.getActiveStrategies).toHaveBeenCalledTimes(2);
  });
});
