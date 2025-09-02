import { default as api } from './api';
import type { ApiResponse } from '@/types/api.types';

// Safety status interface
export interface SafetyStatus {
  tradingMode: 'live' | 'paper';
  emergencyStopActive: boolean;
  circuitBreakers: {
    active: boolean;
    reason?: string;
    triggeredAt?: string;
    maxDailyLoss?: number;
    currentDailyLoss?: number;
    maxTradesPerDay?: number;
    currentTradeCount?: number;
  };
  cooldowns: {
    active: boolean;
    endsAt?: string;
    remainingSeconds?: number;
    reason?: string;
  };
}

// Circuit breaker configuration
export interface CircuitBreakerConfig {
  enabled: boolean;
  maxDailyLoss: number;
  maxDrawdownPercent: number;
  maxTradesPerDay: number;
  maxConsecutiveLosses: number;
}

// Cooldown configuration
export interface CooldownConfig {
  enabled: boolean;
  durationSeconds: number;
  afterConsecutiveLosses: number;
  afterMaxDrawdown: boolean;
}

// Risk limits configuration
export interface RiskLimitsConfig {
  maxPositionSizePercent: number;
  maxExposurePerSectorPercent: number;
  maxExposurePerAssetClassPercent: number;
}

// These types are exported directly with the interface declarations above

// Safety API service
export const safetyApi = {
  // Get current safety status
  getSafetyStatus: () => 
    apiRequest<SafetyStatus>({ 
      url: '/safety/status', 
      method: 'GET' 
    }),
  
  // Toggle emergency stop
  setEmergencyStop: (active: boolean) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/emergency-stop', 
      method: 'POST', 
      data: { active } 
    }),
  
  // Set trading mode
  setTradingMode: (mode: 'live' | 'paper') => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/trading-mode', 
      method: 'POST', 
      data: { mode } 
    }),
  
  // Get circuit breaker configuration
  getCircuitBreakerConfig: () => 
    apiRequest<CircuitBreakerConfig>({ 
      url: '/safety/circuit-breakers/config', 
      method: 'GET' 
    }),
  
  // Update circuit breaker configuration
  updateCircuitBreakerConfig: (config: CircuitBreakerConfig) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/circuit-breakers/config', 
      method: 'PUT', 
      data: config 
    }),
  
  // Get cooldown configuration
  getCooldownConfig: () => 
    apiRequest<CooldownConfig>({ 
      url: '/safety/cooldowns/config', 
      method: 'GET' 
    }),
  
  // Update cooldown configuration
  updateCooldownConfig: (config: CooldownConfig) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/cooldowns/config', 
      method: 'PUT', 
      data: config 
    }),
  
  // Get risk limits configuration
  getRiskLimitsConfig: () => 
    apiRequest<RiskLimitsConfig>({ 
      url: '/safety/risk-limits/config', 
      method: 'GET' 
    }),
  
  // Update risk limits configuration
  updateRiskLimitsConfig: (config: RiskLimitsConfig) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/risk-limits/config', 
      method: 'PUT', 
      data: config 
    }),
  
  // Reset circuit breaker (manual override)
  resetCircuitBreaker: () => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/circuit-breakers/reset', 
      method: 'POST' 
    }),
  
  // Reset cooldown (manual override)
  resetCooldown: () => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/cooldowns/reset', 
      method: 'POST' 
    }),
  
  // Get safety event log
  getSafetyEvents: (limit = 50) => 
    apiRequest<Array<{
      id: string,
      type: 'emergency_stop' | 'circuit_breaker' | 'cooldown' | 'mode_change',
      action: 'activated' | 'deactivated' | 'triggered' | 'reset',
      timestamp: string,
      reason?: string,
      details?: Record<string, any>
    }>>({ 
      url: `/safety/events?limit=${limit}`, 
      method: 'GET' 
    }),
};

export default safetyApi;
