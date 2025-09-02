import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import BrokerPerformance from '../../src/components/broker/BrokerPerformance';
import { brokerApi } from '../../src/services/brokerApi';
import { toast } from 'react-hot-toast';

// Mock types for TypeScript
type BrokerPerformanceMetrics = {
  pnl: number;
  fillRate: number;
  slippage: number;
  latency: number;
  errorRate: number;
  uptime: number;
  status: string;
  isActive: boolean;
};

type BrokerPerformanceData = {
  [brokerName: string]: BrokerPerformanceMetrics;
};

// Mock dependencies
jest.mock('../../src/services/brokerApi', () => ({
  brokerApi: {
    getBrokerPerformance: jest.fn(),
    triggerFailover: jest.fn(),
    getRiskAlerts: jest.fn(),
    acknowledgeAlert: jest.fn()
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

describe('BrokerPerformance Component', () => {
  const mockData = {
    'Primary Broker (Tradier)': {
      pnl: 1250.75,
      fillRate: 0.98,
      slippage: 0.12,
      latency: 42,
      errorRate: 0.02,
      uptime: 0.999,
      status: 'online',
      isActive: true
    },
    'Secondary Broker (Alpaca)': {
      pnl: 950.25,
      fillRate: 0.96,
      slippage: 0.18,
      latency: 48,
      errorRate: 0.04,
      uptime: 0.995,
      status: 'online',
      isActive: false
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock the broker API
    (brokerApi.getBrokerPerformance as jest.Mock).mockResolvedValue({
      success: true,
      data: mockData
    });
  });

  test('renders all broker cards with correct data', async () => {
    render(<BrokerPerformance />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Primary Broker (Tradier)')).toBeInTheDocument();
    });
    
    // Check broker data is displayed correctly
    expect(screen.getByText('$1250.75')).toBeInTheDocument();
    expect(screen.getByText('98.0%')).toBeInTheDocument(); // Fill Rate
    expect(screen.getByText('Secondary Broker (Alpaca)')).toBeInTheDocument();
    
    // Check status badges
    expect(screen.getAllByText('Online')[0]).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
  });

  test('shows loading state initially', () => {
    render(<BrokerPerformance />, { wrapper: createWrapper() });
    
    // Check loading skeleton is displayed
    expect(screen.getByText('Broker Intelligence')).toBeInTheDocument();
    expect(document.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
  });

  test('handles failover correctly', async () => {
    // Mock the failover API
    (brokerApi.triggerFailover as jest.Mock).mockResolvedValue({
      success: true,
      data: { 
        success: true, 
        activeBroker: 'Secondary Broker (Alpaca)' 
      }
    });

    render(<BrokerPerformance />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Primary Broker (Tradier)')).toBeInTheDocument();
    });
    
    // Click failover button
    const failoverButton = screen.getByText('Failover');
    fireEvent.click(failoverButton);
    
    // Verify API was called with correct parameters
    await waitFor(() => {
      expect(brokerApi.triggerFailover).toHaveBeenCalledWith(
        'Primary Broker (Tradier)',
        'Secondary Broker (Alpaca)'
      );
    });
    
    // Check toast was shown
    expect(toast.success).toHaveBeenCalledWith(
      expect.stringContaining('Successfully failed over to Secondary Broker (Alpaca)')
    );
  });

  test('handles error state correctly', async () => {
    // Mock API error
    (brokerApi.getBrokerPerformance as jest.Mock).mockResolvedValue({
      success: false,
      error: 'Failed to fetch broker data'
    });

    render(<BrokerPerformance />, { wrapper: createWrapper() });
    
    // Wait for error state
    await waitFor(() => {
      expect(screen.getByText('Could not load broker performance data')).toBeInTheDocument();
    });
    
    // Check retry button exists
    const retryButton = screen.getByText('Retry');
    expect(retryButton).toBeInTheDocument();
    
    // Reset mock
    (brokerApi.getBrokerPerformance as jest.Mock).mockResolvedValue({
      success: true,
      data: mockData
    });
    
    // Click retry
    fireEvent.click(retryButton);
    
    // Verify API was called again
    expect(brokerApi.getBrokerPerformance).toHaveBeenCalledTimes(2);
  });

  test('refresh button refetches data', async () => {
    render(<BrokerPerformance />, { wrapper: createWrapper() });
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Primary Broker (Tradier)')).toBeInTheDocument();
    });
    
    // Find and click refresh button
    const refreshButton = document.querySelector('button[aria-label="Refresh"]') || 
                         document.querySelector('button.absolute.top-2.right-2');
    fireEvent.click(refreshButton as HTMLElement);
    
    // Verify toast and refetch
    expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('Refreshing broker data'));
    expect(brokerApi.getBrokerPerformance).toHaveBeenCalledTimes(2);
  });
});
