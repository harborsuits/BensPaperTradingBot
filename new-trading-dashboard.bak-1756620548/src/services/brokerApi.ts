import api from './api';
import { AxiosRequestConfig } from 'axios';

// Direct API request function
async function apiRequest<T>(config: AxiosRequestConfig) {
  try {
    const response = await fetch(config.url as string, {
      method: config.method,
      headers: {
        'Content-Type': 'application/json',
        ...(config.headers || {})
      },
      body: config.data ? JSON.stringify(config.data) : undefined
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return { success: true, data };
  } catch (error: any) {
    console.error('API request failed:', error);
    return { success: false, error: error.message || 'Unknown error' };
  }
};

export interface BrokerPerformanceMetrics {
  pnl: number;
  fillRate: number;
  slippage: number;
  latency: number;
  errorRate: number;
  uptime: number;
  status: 'online' | 'degraded' | 'offline' | 'circuit_broken';
  isActive: boolean;
}

export interface BrokerPerformanceData {
  [broker: string]: BrokerPerformanceMetrics;
}

// Broker API methods
export const brokerApi = {
  // Get broker performance metrics
  async getBrokerPerformance() {
    return apiRequest<BrokerPerformanceData>({ 
      url: '/api/broker/performance', 
      method: 'GET' 
    });
  },
  
  // Trigger broker failover
  async triggerFailover(fromBroker: string, toBroker: string) {
    return apiRequest<{ success: boolean; activeBroker: string }>({ 
      url: '/api/broker/failover', 
      method: 'POST',
      data: { fromBroker, toBroker }
    });
  },
  
  // Get broker trade metrics
  async getBrokerTradeMetrics(brokerId: string, timeframe: 'day' | 'week' | 'month' = 'day') {
    return apiRequest<any>({ 
      url: `/api/broker/${brokerId}/metrics?timeframe=${timeframe}`, 
      method: 'GET' 
    });
  },
  
  // Get risk alerts
  async getRiskAlerts() {
    return apiRequest<any>({ 
      url: '/api/alerts', 
      method: 'GET' 
    });
  },
  
  // Acknowledge risk alert
  async acknowledgeAlert(alertId: string) {
    return apiRequest<{ success: boolean }>({ 
      url: `/api/alerts/${alertId}/acknowledge`, 
      method: 'POST' 
    });
  }
};
