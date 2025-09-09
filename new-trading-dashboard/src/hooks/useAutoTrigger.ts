import { useState, useEffect, useCallback } from 'react';
import autoTriggerService, { TriggerRule, AutoTriggerConfig, MarketCondition } from '@/services/autoTriggerService';

interface UseAutoTriggerReturn {
  rules: TriggerRule[];
  config: AutoTriggerConfig;
  isMonitoring: boolean;
  lastCheck: Date | null;
  activeTriggers: string[];
  triggerStats: any[];

  // Actions
  startMonitoring: () => void;
  stopMonitoring: () => void;
  updateRule: (ruleId: string, updates: Partial<TriggerRule>) => void;
  updateConfig: (config: Partial<AutoTriggerConfig>) => void;
  resetTriggerCooldown: (ruleId: string) => void;
  getTriggerEvents: () => TriggerEvent[];
}

interface TriggerEvent {
  id: string;
  type: string;
  ruleId: string;
  sessionId?: string;
  marketData?: MarketCondition;
  timestamp: string;
}

export const useAutoTrigger = (): UseAutoTriggerReturn => {
  const [rules, setRules] = useState<TriggerRule[]>([]);
  const [config, setConfig] = useState<AutoTriggerConfig>(autoTriggerService.getConfig());
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  const [activeTriggers, setActiveTriggers] = useState<string[]>([]);
  const [triggerEvents, setTriggerEvents] = useState<TriggerEvent[]>([]);

  // Load initial data
  useEffect(() => {
    setRules(autoTriggerService.getRules());
    setConfig(autoTriggerService.getConfig());
    setActiveTriggers(autoTriggerService.getActiveTriggers());
  }, []);

  // Listen for trigger events
  useEffect(() => {
    const handleTriggerEvent = (event: CustomEvent) => {
      const { type, data } = event.detail;
      const triggerEvent: TriggerEvent = {
        id: `${type}_${Date.now()}`,
        type,
        ...data,
        timestamp: new Date().toISOString()
      };

      setTriggerEvents(prev => [triggerEvent, ...prev.slice(0, 49)]); // Keep last 50 events

      // Update active triggers
      setActiveTriggers(autoTriggerService.getActiveTriggers());

      // Refresh rules data
      setRules(autoTriggerService.getRules());
    };

    window.addEventListener('evo_trigger_event', handleTriggerEvent as EventListener);

    return () => {
      window.removeEventListener('evo_trigger_event', handleTriggerEvent as EventListener);
    };
  }, []);

  // Periodic updates
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveTriggers(autoTriggerService.getActiveTriggers());
      setRules(autoTriggerService.getRules());
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const startMonitoring = useCallback(() => {
    autoTriggerService.startMonitoring();
    setIsMonitoring(true);
  }, []);

  const stopMonitoring = useCallback(() => {
    autoTriggerService.stopMonitoring();
    setIsMonitoring(false);
  }, []);

  const updateRule = useCallback((ruleId: string, updates: Partial<TriggerRule>) => {
    autoTriggerService.updateRule(ruleId, updates);
    setRules(autoTriggerService.getRules());
  }, []);

  const updateConfig = useCallback((configUpdates: Partial<AutoTriggerConfig>) => {
    autoTriggerService.updateConfig(configUpdates);
    setConfig(autoTriggerService.getConfig());
  }, []);

  const resetTriggerCooldown = useCallback((ruleId: string) => {
    autoTriggerService.resetTriggerCooldown(ruleId);
    setRules(autoTriggerService.getRules());
  }, []);

  const getTriggerEvents = useCallback(() => {
    return triggerEvents;
  }, [triggerEvents]);

  return {
    rules,
    config,
    isMonitoring,
    lastCheck,
    activeTriggers,
    triggerStats: autoTriggerService.getTriggerStats(),

    // Actions
    startMonitoring,
    stopMonitoring,
    updateRule,
    updateConfig,
    resetTriggerCooldown,
    getTriggerEvents
  };
};

export default useAutoTrigger;
