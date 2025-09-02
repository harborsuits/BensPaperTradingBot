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

  return null;
};

export default useEvoTesterWebSocket;
