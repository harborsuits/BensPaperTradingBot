import { useQueryClient } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';
import { createToast } from '@/components/ui/Toast';

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
};

export type ChatUpdate = {
  messageId: string;
  type: 'new' | 'update' | 'thinking';
  message: ChatMessage;
};

/**
 * Hook to handle WebSocket messages for AI Chat functionality
 */
export const useAIChatWebSocket = () => {
  const queryClient = useQueryClient();
  
  // Handle new chat messages and updates
  useWebSocketMessage<ChatUpdate>('ai_chat_message', (message) => {
    const chatUpdate = message.data;
    
    // Invalidate queries to trigger refetch of chat history
    queryClient.invalidateQueries(['ai', 'chat', 'history']);
    
    // Show toast for important messages based on content or other criteria
    if (chatUpdate.type === 'new' && chatUpdate.message.role === 'assistant') {
      // Check if message contains alert keywords
      const alertKeywords = ['alert', 'warning', 'important', 'attention'];
      const containsAlert = alertKeywords.some(keyword => 
        chatUpdate.message.content.toLowerCase().includes(keyword)
      );
      
      if (containsAlert) {
        createToast({
          title: 'Trading Assistant Alert',
          description: `Important message from BenBot: "${chatUpdate.message.content.substring(0, 100)}${chatUpdate.message.content.length > 100 ? '...' : ''}"`,
          variant: 'warning',
          duration: 7000,
        });
      }
    }
  });
  
  // Handle trading context updates relevant to the chat
  useWebSocketMessage<{contextType: string, data: any}>('trading_context_update', (message) => {
    const contextUpdate = message.data;
    
    // Update AI context in cache
    queryClient.setQueryData(['ai', 'context', contextUpdate.contextType], contextUpdate.data);
    
    // Optionally notify user of significant context changes
    if (contextUpdate.contextType === 'market_regime_change' || 
        contextUpdate.contextType === 'high_priority_alert') {
      createToast({
        title: 'Trading Context Update',
        description: `Your AI assistant has new information about ${contextUpdate.contextType}`,
        variant: 'info',
        duration: 5000,
      });
    }
  });

  return null;
};

export default useAIChatWebSocket;
