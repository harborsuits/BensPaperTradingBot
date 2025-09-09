import { useQueryClient } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';
import { EvoTesterProgress, EvoStrategy } from '@/types/api.types';
import { createToast } from '@/components/ui/Toast';

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
      createToast({
        title: 'Evolution Completed',
        description: `Session ${progressData.sessionId} has completed after ${progressData.currentGeneration} generations.`,
        variant: 'success',
        duration: 5000,
        action: {
          label: 'View Results',
          onClick: () => {
            window.location.href = `/evotester/results/${progressData.sessionId}`;
          }
        }
      });
    } else if (progressData.status === 'failed') {
      createToast({
        title: 'Evolution Failed',
        description: progressData.errorMessage || 'The evolution process failed to complete.',
        variant: 'destructive',
        duration: 7000,
      });
    }
  });
  
  // Handle new strategy notifications
  useWebSocketMessage<EvoStrategy>('evotester_new_strategy', (message) => {
    const strategy = message.data;
    
    createToast({
      title: 'New Evolved Strategy',
      description: `A new strategy "${strategy.name}" has been evolved with fitness ${strategy.fitness.toFixed(3)}.`,
      variant: 'info',
      duration: 5000,
      action: {
        label: 'Inspect',
        onClick: () => {
          // This would typically open a modal or navigate to a details page
          queryClient.setQueryData(['evoTester', 'strategy', strategy.id], strategy);
          window.dispatchEvent(new CustomEvent('openStrategyInspector', { detail: strategy }));
        }
      }
    });
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
      createToast({
        title: 'Evolution Paused',
        description: statusData.message || `Session ${statusData.sessionId} has been paused.`,
        variant: 'warning',
        duration: 3000,
      });
    } else if (statusData.status === 'running' && statusData.message?.includes('resumed')) {
      createToast({
        title: 'Evolution Resumed',
        description: `Session ${statusData.sessionId} has been resumed.`,
        variant: 'info',
        duration: 3000,
      });
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

    createToast({
      title: 'Auto-Trigger Activated',
      description: `Evolution experiment started for trigger rule "${triggerData.ruleId}".`,
      variant: 'info',
      duration: 4000,
      action: {
        label: 'View Experiment',
        onClick: () => {
          queryClient.invalidateQueries(['evoTester', 'sessions']);
          // Could navigate to the specific experiment
        }
      }
    });

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

    createToast({
      title: promotionData.success ? 'Strategy Promoted!' : 'Strategy Rejected',
      description: `Strategy "${promotionData.candidateName}" has been ${
        promotionData.success ? 'promoted to competition' : 'rejected during validation'
      }.`,
      variant: promotionData.success ? 'success' : 'warning',
      duration: 5000,
      action: promotionData.success ? {
        label: 'View in Competition',
        onClick: () => {
          // Could navigate to competition view or strategy details
          window.dispatchEvent(new CustomEvent('strategy_promoted', {
            detail: promotionData
          }));
        }
      } : undefined
    });

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

    createToast({
      title: 'Capital Update',
      description: `${messages[allocationData.type]}: $${allocationData.amount.toFixed(2)}`,
      variant: allocationData.type === 'emergency_stop' ? 'destructive' :
               allocationData.type === 'allocated' ? 'info' : 'success',
      duration: 3000
    });

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

      createToast({
        title: 'Evolution Milestone',
        description: descriptions[milestoneData.milestone as keyof typeof descriptions] ||
                    `Generation ${milestoneData.generation} completed`,
        variant: 'info',
        duration: 3000
      });
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

    createToast({
      title: 'Market Condition Alert',
      description: alertData.message,
      variant: alertData.severity === 'high' ? 'destructive' :
               alertData.severity === 'medium' ? 'warning' : 'info',
      duration: alertData.severity === 'high' ? 7000 : 5000,
      action: alertData.recommendation ? {
        label: 'View Recommendation',
        onClick: () => {
          createToast({
            title: 'Recommendation',
            description: alertData.recommendation,
            variant: 'info',
            duration: 8000
          });
        }
      } : undefined
    });
  });

  return null;
};

export default useEvoTesterWebSocket;
