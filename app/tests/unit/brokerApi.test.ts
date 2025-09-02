import { brokerApi } from '../../src/services/brokerApi';
import fetchMock from 'jest-fetch-mock';

// Mock apiRequest function that's used internally by broker API
jest.mock('../../src/services/api', () => ({
  apiRequest: jest.fn((config) => {
    return fetch(config.url, {
      method: config.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: config.data ? JSON.stringify(config.data) : undefined
    }).then(async (res) => {
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      return { success: true, data };
    }).catch((error) => ({
      success: false,
      error: error.message || 'Unknown error'
    }));
  })
}));

// Mock fetch for testing
fetchMock.enableMocks();

describe('Broker API Service', () => {
  beforeEach(() => {
    fetchMock.resetMocks();
  });

  test('getBrokerPerformance should return broker performance data', async () => {
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
      }
    };

    fetchMock.mockResponseOnce(JSON.stringify(mockData));

    const response = await brokerApi.getBrokerPerformance();
    
    expect(fetchMock).toHaveBeenCalledWith('/api/broker/performance', expect.any(Object));
    expect(response.success).toBe(true);
    expect(response.data).toEqual(mockData);
  });

  test('getBrokerPerformance should handle errors', async () => {
    fetchMock.mockRejectOnce(new Error('Network error'));

    const response = await brokerApi.getBrokerPerformance();
    
    expect(response.success).toBe(false);
    expect(response.error).toBe('Network error');
  });

  test('triggerFailover should call correct endpoint with params', async () => {
    const mockResponse = { 
      success: true, 
      activeBroker: 'Secondary Broker (Alpaca)' 
    };

    fetchMock.mockResponseOnce(JSON.stringify(mockResponse));

    const response = await brokerApi.triggerFailover(
      'Primary Broker (Tradier)', 
      'Secondary Broker (Alpaca)'
    );
    
    expect(fetchMock).toHaveBeenCalledWith('/api/broker/failover', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        fromBroker: 'Primary Broker (Tradier)',
        toBroker: 'Secondary Broker (Alpaca)'
      }),
    });
    
    expect(response.success).toBe(true);
    expect(response.data.activeBroker).toBe('Secondary Broker (Alpaca)');
  });

  test('getRiskAlerts should fetch alerts with correct parameters', async () => {
    const mockAlerts = [
      {
        id: 'alert-1',
        type: 'DRAWDOWN',
        message: 'Portfolio drawdown exceeded 5% threshold',
        severity: 'HIGH',
        timestamp: '2025-05-11T12:00:00Z',
        source: 'risk_monitor',
        acknowledged: false
      }
    ];

    fetchMock.mockResponseOnce(JSON.stringify(mockAlerts));

    const response = await brokerApi.getRiskAlerts();
    
    expect(fetchMock).toHaveBeenCalledWith('/api/alerts', expect.any(Object));
    expect(response.success).toBe(true);
    expect(response.data).toEqual(mockAlerts);
  });

  test('acknowledgeAlert should call correct endpoint', async () => {
    const mockResponse = { success: true };
    fetchMock.mockResponseOnce(JSON.stringify(mockResponse));

    const response = await brokerApi.acknowledgeAlert('alert-123');
    
    expect(fetchMock).toHaveBeenCalledWith('/api/alerts/alert-123/acknowledge', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: undefined,
    });
    
    expect(response.success).toBe(true);
  });
});
