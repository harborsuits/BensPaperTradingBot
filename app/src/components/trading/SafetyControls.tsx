import React, { useState, useEffect } from 'react';
import { ShieldAlert, Clock, ToggleLeft, ToggleRight, Ban } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';
import { portfolioApi } from '@/services/api';

interface SafetyStatus {
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

const SafetyControls: React.FC = () => {
  const queryClient = useQueryClient();
  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus>({
    tradingMode: 'paper',
    emergencyStopActive: false,
    circuitBreakers: {
      active: false
    },
    cooldowns: {
      active: false
    }
  });
  
  // Countdown timer for cooldown
  const [cooldownTimeLeft, setCooldownTimeLeft] = useState<number>(0);
  
  // Get initial safety status
  useQuery({
    queryKey: ['safetyStatus'],
    queryFn: () => portfolioApi.getSafetyStatus(),
    onSuccess: (response) => {
      if (!response?.success) return;
      const d = (response as any).data || {};
      const merged: SafetyStatus = {
        tradingMode: d.tradingMode ?? 'paper',
        emergencyStopActive: !!d.emergencyStopActive,
        circuitBreakers: {
          active: !!d.circuitBreakers?.active,
          reason: d.circuitBreakers?.reason,
          triggeredAt: d.circuitBreakers?.triggeredAt,
          maxDailyLoss: d.circuitBreakers?.maxDailyLoss,
          currentDailyLoss: d.circuitBreakers?.currentDailyLoss,
          maxTradesPerDay: d.circuitBreakers?.maxTradesPerDay,
          currentTradeCount: d.circuitBreakers?.currentTradeCount,
        },
        cooldowns: {
          active: !!d.cooldowns?.active,
          endsAt: d.cooldowns?.endsAt,
          remainingSeconds: d.cooldowns?.remainingSeconds ?? 0,
          reason: d.cooldowns?.reason,
        },
      };
      setSafetyStatus(merged);
      if (merged.cooldowns.active && merged.cooldowns.remainingSeconds) {
        setCooldownTimeLeft(merged.cooldowns.remainingSeconds);
      }
    }
  });
  
  // Mutations
  const toggleTradingMode = useMutation({
    mutationFn: (mode: 'live' | 'paper') => portfolioApi.setTradingMode(mode),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['safetyStatus'] });
    }
  });
  
  const toggleEmergencyStop = useMutation({
    mutationFn: (active: boolean) => portfolioApi.setEmergencyStop(active),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['safetyStatus'] });
    }
  });
  
  // WebSocket updates
  useWebSocketMessage<SafetyStatus>(
    'safety_status_update',
    (message) => {
      setSafetyStatus(message.data);
      
      if (message.data.cooldowns.active && message.data.cooldowns.remainingSeconds) {
        setCooldownTimeLeft(message.data.cooldowns.remainingSeconds);
      }
    },
    []
  );
  
  // Countdown timer effect
  useEffect(() => {
    if (!safetyStatus.cooldowns.active || cooldownTimeLeft <= 0) return;
    
    const timer = setInterval(() => {
      setCooldownTimeLeft(prev => {
        const newValue = prev - 1;
        if (newValue <= 0) {
          clearInterval(timer);
          return 0;
        }
        return newValue;
      });
    }, 1000);
    
    return () => clearInterval(timer);
  }, [safetyStatus.cooldowns.active, cooldownTimeLeft]);
  
  // Format time remaining for cooldown
  const formatTimeRemaining = (seconds: number): string => {
    if (seconds <= 0) return '00:00';
    
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Emergency Stop */}
      <div 
        className={`rounded-lg border p-4 flex flex-col items-center justify-center
          ${safetyStatus.emergencyStopActive 
            ? 'bg-bear/10 border-bear' 
            : 'bg-card border-border'
          }`}
      >
        <div className="flex items-center justify-between w-full mb-2">
          <h3 className="font-medium">Kill Switch</h3>
          <button
            onClick={() => toggleEmergencyStop.mutate(!safetyStatus.emergencyStopActive)}
            className={`rounded-full p-1 transition-colors
              ${safetyStatus.emergencyStopActive
                ? 'bg-bear/20 text-bear hover:bg-bear/30'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            aria-label={safetyStatus.emergencyStopActive ? 'Deactivate emergency stop' : 'Activate emergency stop'}
          >
            <Ban size={24} />
          </button>
        </div>
        
        <div className="w-full">
          <div className="flex items-center justify-between">
            <span className="text-sm">Status:</span>
            <span 
              className={`text-sm font-medium
                ${safetyStatus.emergencyStopActive 
                  ? 'text-bear' 
                  : 'text-success'
                }`}
            >
              {safetyStatus.emergencyStopActive ? 'ACTIVE' : 'READY'}
            </span>
          </div>
          
          <p className="text-xs text-muted-foreground mt-2">
            {safetyStatus.emergencyStopActive 
              ? 'All trading is currently stopped. Click the button to resume.'
              : 'Click the button to immediately stop all trading activity.'
            }
          </p>
        </div>
      </div>
      
      {/* Trading Mode Toggle */}
      <div className="rounded-lg border border-border bg-card p-4 flex flex-col">
        <div className="flex items-center justify-between w-full mb-2">
          <h3 className="font-medium">Trading Mode</h3>
          <button
            onClick={() => toggleTradingMode.mutate(safetyStatus.tradingMode === 'live' ? 'paper' : 'live')}
            className="p-1 rounded-full hover:bg-muted/80"
            aria-label={`Switch to ${safetyStatus.tradingMode === 'live' ? 'paper' : 'live'} trading`}
          >
            {safetyStatus.tradingMode === 'live' 
              ? <ToggleRight size={24} className="text-success" /> 
              : <ToggleLeft size={24} className="text-muted-foreground" />
            }
          </button>
        </div>
        
        <div className="w-full">
          <div className="flex items-center justify-between">
            <span className="text-sm">Current Mode:</span>
            <span 
              className={`text-sm font-medium
                ${safetyStatus.tradingMode === 'live' 
                  ? 'text-success' 
                  : 'text-primary'
                }`}
            >
              {safetyStatus.tradingMode === 'live' ? 'LIVE' : 'PAPER'}
            </span>
          </div>
          
          <p className="text-xs text-muted-foreground mt-2">
            {safetyStatus.tradingMode === 'live' 
              ? 'System is using real money for trades. Toggle to switch to paper trading.'
              : 'System is in sandbox mode (paper trading). Toggle to switch to live trading.'
            }
          </p>
        </div>
      </div>
      
      {/* Circuit Breaker Status */}
      <div 
        className={`rounded-lg border p-4 flex flex-col
          ${safetyStatus.circuitBreakers.active 
            ? 'bg-highImpact/10 border-highImpact' 
            : 'bg-card border-border'
          }`}
      >
        <div className="flex items-center mb-2">
          <ShieldAlert 
            size={20} 
            className={`mr-2 ${safetyStatus.circuitBreakers.active ? 'text-highImpact' : 'text-muted-foreground'}`} 
          />
          <h3 className="font-medium">Circuit Breaker</h3>
        </div>
        
        <div className="w-full">
          <div className="flex items-center justify-between">
            <span className="text-sm">Status:</span>
            <span 
              className={`text-sm font-medium
                ${safetyStatus.circuitBreakers.active 
                  ? 'text-highImpact' 
                  : 'text-success'
                }`}
            >
              {safetyStatus.circuitBreakers.active ? 'TRIGGERED' : 'NORMAL'}
            </span>
          </div>
          
          {safetyStatus.circuitBreakers.active && (
            <>
              <div className="flex items-center justify-between mt-1">
                <span className="text-sm">Reason:</span>
                <span className="text-sm">{safetyStatus.circuitBreakers.reason}</span>
              </div>
              
              {safetyStatus.circuitBreakers.maxDailyLoss && (
                <div className="flex items-center justify-between mt-1">
                  <span className="text-sm">Daily Loss:</span>
                  <span className="text-sm">${safetyStatus.circuitBreakers.currentDailyLoss?.toFixed(2)} / ${safetyStatus.circuitBreakers.maxDailyLoss.toFixed(2)}</span>
                </div>
              )}
              
              {safetyStatus.circuitBreakers.maxTradesPerDay && (
                <div className="flex items-center justify-between mt-1">
                  <span className="text-sm">Trade Count:</span>
                  <span className="text-sm">{safetyStatus.circuitBreakers.currentTradeCount} / {safetyStatus.circuitBreakers.maxTradesPerDay}</span>
                </div>
              )}
            </>
          )}
          
          {!safetyStatus.circuitBreakers.active && (
            <p className="text-xs text-muted-foreground mt-2">
              Circuit breakers are active and monitoring for excessive losses or trades.
            </p>
          )}
        </div>
      </div>
      
      {/* Cooldown Timer */}
      <div 
        className={`rounded-lg border p-4 flex flex-col
          ${safetyStatus.cooldowns.active 
            ? 'bg-infoImpact/10 border-infoImpact' 
            : 'bg-card border-border'
          }`}
      >
        <div className="flex items-center mb-2">
          <Clock 
            size={20} 
            className={`mr-2 ${safetyStatus.cooldowns.active ? 'text-infoImpact' : 'text-muted-foreground'}`} 
          />
          <h3 className="font-medium">Trading Cooldown</h3>
        </div>
        
        <div className="w-full">
          <div className="flex items-center justify-between">
            <span className="text-sm">Status:</span>
            <span 
              className={`text-sm font-medium
                ${safetyStatus.cooldowns.active 
                  ? 'text-infoImpact' 
                  : 'text-success'
                }`}
            >
              {safetyStatus.cooldowns.active ? 'ACTIVE' : 'READY'}
            </span>
          </div>
          
          {safetyStatus.cooldowns.active && (
            <>
              <div className="flex items-center justify-between mt-1">
                <span className="text-sm">Remaining:</span>
                <span className="text-sm font-mono">{formatTimeRemaining(cooldownTimeLeft)}</span>
              </div>
              
              <div className="flex items-center justify-between mt-1">
                <span className="text-sm">Reason:</span>
                <span className="text-sm">{safetyStatus.cooldowns.reason}</span>
              </div>
            </>
          )}
          
          {!safetyStatus.cooldowns.active && (
            <p className="text-xs text-muted-foreground mt-2">
              No active cooldown. System will enter cooldown after losses to prevent overtrading.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default SafetyControls;
