import { useQueryClient } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';
import { EvoTesterProgress, EvoStrategy } from '@/types/api.types';
import { showInfoToast, showSuccessToast, showErrorToast, showWarningToast } from '@/utils/toast';

/**
 * Hook to handle WebSocket messages for EvoTester progress and updates
 * @param sessionId Optional session ID to filter updates for a specific session
 */
export const useEvoTesterWebSocket = (sessionId?: string) => {
  const queryClient = useQueryClient();
  
  // Handle progress updates
  useWebSocketMessage<EvoTesterProgress>('evotester_progress', (message) => {
    const progressData = message.data;
    
    // If sessionId is provided, only process updates for that session
    if (sessionId && progressData.sessionId !== sessionId) {
      return;
    }
    
    // Invalidate queries to trigger refetch
    queryClient.invalidateQueries(['evoTester', 'status', progressData.sessionId]);
    
    // If we're not looking at a specific session, also invalidate the list
    if (!sessionId) {
      queryClient.invalidateQueries(['evoTester', 'history']);
    }
    
    // If the status changed to completed/failed, show a toast notification
    if (progressData.status === 'completed') {
      showSuccessToast(`Session ${progressData.sessionId} has completed after ${progressData.currentGeneration} generations.`);
    } else if (progressData.status === 'failed') {
      showErrorToast(progressData.errorMessage || 'The evolution process failed to complete.');
    }
  });

  // New schema messages for real-mode proof
  useWebSocketMessage<any>('session_started', (message) => {
    showInfoToast(`Session ${message.data.session_id} mode=${message.data.mode}, data_source=${message.data.data_source}`);
  });

  useWebSocketMessage<any>('gen_complete', (message) => {
    // Lightweight toast for first few gens only to avoid noise
    const g = Number(message.data?.g || 0);
    if (g <= 3) {
      showInfoToast(`Generation ${g} complete - Best OOS Sharpe: ${message.data?.best?.oos_sharpe ?? 'n/a'}`);
    }
  });

  useWebSocketMessage<any>('checkpoint', (message) => {
    // No UI noise; could drive a progress bar if needed
  });

  useWebSocketMessage<any>('session_completed', (message) => {
    showSuccessToast(`Evolution Completed - Best strategy: ${message.data?.best_strategy_id}`);
  });

  useWebSocketMessage<any>('error', (message) => {
    showErrorToast(String(message.data?.message || 'Unknown error'));
  });
  
  // Handle new strategy notifications
  useWebSocketMessage<EvoStrategy>('evotester_new_strategy', (message) => {
    const strategy = message.data;
    
    showInfoToast(`A new strategy "${strategy.name}" has been evolved with fitness ${strategy.fitness.toFixed(3)}.`);
    // Store strategy in cache for potential inspection
    queryClient.setQueryData(['evoTester', 'strategy', strategy.id], strategy);
    window.dispatchEvent(new CustomEvent('openStrategyInspector', { detail: strategy }));
  });
  
  // Handle status change notifications
  useWebSocketMessage<{sessionId: string, status: string, message: string}>('evotester_status_change', (message) => {
    const statusData = message.data;

    // If sessionId is provided, only process updates for that session
    if (sessionId && statusData.sessionId !== sessionId) {
      return;
    }

    // Invalidate status query to trigger refetch
    queryClient.invalidateQueries(['evoTester', 'status', statusData.sessionId]);

    // Show a toast for significant status changes
    if (statusData.status === 'paused') {
      showWarningToast(statusData.message || `Session ${statusData.sessionId} has been paused.`);
    } else if (statusData.status === 'running' && statusData.message?.includes('resumed')) {
      showInfoToast(`Session ${statusData.sessionId} has been resumed.`);
    }
  });

  // Handle auto-trigger events
  useWebSocketMessage<{
    ruleId: string;
    sessionId: string;
    marketData: any;
    timestamp: string;
  }>('evo_trigger_activated', (message) => {
    const triggerData = message.data;

    showInfoToast(`Evolution experiment started for trigger rule "${triggerData.ruleId}".`);

    // Invalidate sessions to show the new auto-triggered experiment
    queryClient.invalidateQueries(['evoTester', 'sessions']);
  });

  // Handle promotion events
  useWebSocketMessage<{
    candidateId: string;
    candidateName: string;
    pipelineId: string;
    success: boolean;
    fitness: number;
  }>('strategy_promoted', (message) => {
    const promotionData = message.data;

    if (promotionData.success) {
      showSuccessToast(`Strategy "${promotionData.candidateName}" has been promoted to competition.`);
      window.dispatchEvent(new CustomEvent('strategy_promoted', { detail: promotionData }));
    } else {
      showWarningToast(`Strategy "${promotionData.candidateName}" has been rejected during validation.`);
    }

    // Invalidate relevant queries
    queryClient.invalidateQueries(['strategies']);
    queryClient.invalidateQueries(['competition']);
  });

  // Handle capital allocation events
  useWebSocketMessage<{
    allocationId: string;
    experimentId: string;
    amount: number;
    poolId: string;
    type: 'allocated' | 'released' | 'emergency_stop';
  }>('capital_allocation_update', (message) => {
    const allocationData = message.data;

    const messages = {
      allocated: `Capital allocated to experiment ${allocationData.experimentId.slice(-8)}`,
      released: `Capital released from experiment ${allocationData.experimentId.slice(-8)}`,
      emergency_stop: `Emergency stop triggered for experiment ${allocationData.experimentId.slice(-8)}`
    };

    const notificationMessage = `${messages[allocationData.type]}: $${allocationData.amount.toFixed(2)}`;
    if (allocationData.type === 'emergency_stop') {
      showErrorToast(notificationMessage);
    } else if (allocationData.type === 'allocated') {
      showInfoToast(notificationMessage);
    } else {
      showSuccessToast(notificationMessage);
    }

    // Invalidate capital-related queries
    queryClient.invalidateQueries(['capital']);
    queryClient.invalidateQueries(['pools']);
  });

  // Handle evolution milestone events
  useWebSocketMessage<{
    sessionId: string;
    generation: number;
    milestone: string;
    fitness: number;
    improvement: number;
  }>('evolution_milestone', (message) => {
    const milestoneData = message.data;

    // Only show toast for significant milestones
    if (milestoneData.milestone === 'new_champion' ||
        milestoneData.milestone === 'fitness_improvement' ||
        milestoneData.generation % 10 === 0) {

      const descriptions = {
        new_champion: `New champion strategy evolved with fitness ${milestoneData.fitness.toFixed(3)}`,
        fitness_improvement: `Fitness improved by ${(milestoneData.improvement * 100).toFixed(1)}%`,
        generation_milestone: `Reached generation ${milestoneData.generation}`
      };

      showInfoToast(descriptions[milestoneData.milestone as keyof typeof descriptions] ||
                    `Generation ${milestoneData.generation} completed`);
    }

    // Update progress data
    queryClient.invalidateQueries(['evoTester', 'status', milestoneData.sessionId]);
  });

  // Handle market condition alerts
  useWebSocketMessage<{
    condition: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
    recommendation?: string;
  }>('market_condition_alert', (message) => {
    const alertData = message.data;

    if (alertData.severity === 'high') {
      showErrorToast(alertData.message);
    } else if (alertData.severity === 'medium') {
      showWarningToast(alertData.message);
    } else {
      showInfoToast(alertData.message);
    }
    
    // Show recommendation in console if available
    if (alertData.recommendation) {
      console.log('Market Recommendation:', alertData.recommendation);
    }
  });

  return null;
};

export default useEvoTesterWebSocket;
