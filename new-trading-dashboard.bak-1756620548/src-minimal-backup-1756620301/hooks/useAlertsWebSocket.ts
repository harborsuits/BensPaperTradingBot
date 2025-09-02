// We don't need to import useEffect for this hook
import { useQueryClient } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';
import { Alert } from '@/types/alerts.types';
import { createToast } from '@/components/ui/Toast'; // This will be our notification toast component

/**
 * Hook to handle WebSocket alerts and notifications
 */
export const useAlertsWebSocket = () => {
  const queryClient = useQueryClient();
  
  // Handle new alert messages from WebSocket
  useWebSocketMessage<Alert>('new_alert', (message) => {
    const alert = message.data;
    // Invalidate the alerts queries to trigger a refetch
    queryClient.invalidateQueries(['alerts']);
    queryClient.invalidateQueries(['alertsState']);
    
    // Show a toast notification for the new alert
    const severity = alert.severity === 'error' ? 'destructive' : 
                     alert.severity === 'warning' ? 'warning' : 
                     alert.severity === 'success' ? 'success' : 'default';
                     
    createToast({
      title: alert.title,
      description: alert.message,
      variant: severity,
      action: alert.actionRequired ? {
        label: 'View',
        onClick: () => {
          if (alert.actionUrl) {
            window.location.href = alert.actionUrl;
          } else {
            window.location.href = `/alerts/${alert.id}`;
          }
        }
      } : undefined
    });
  });
  
  // Handle alert status updates
  useWebSocketMessage<{ id: string, status: string }>('alert_status_change', (message) => {
    const data = message.data;
    // Update the specific alert in the query cache
    queryClient.setQueryData(['alert', data.id], (oldData: Alert | undefined) => {
      if (!oldData) return oldData;
      return {
        ...oldData,
        status: data.status as Alert['status']
      };
    });
    
    // Also invalidate the alerts list
    queryClient.invalidateQueries(['alerts']);
    queryClient.invalidateQueries(['alertsState']);
  });

  return null;
};

export default useAlertsWebSocket;
