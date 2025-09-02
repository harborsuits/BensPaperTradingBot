import React from 'react';
import { User, Bot } from 'lucide-react';
import { ChatMessage } from '@/hooks/useAIChatWebSocket';

// Helper function to combine class names
const cn = (...classes: (string | undefined | null | false)[]) => {
  return classes.filter(Boolean).join(' ');
};

// Simple formatter for timestamp
const formatTime = (timestamp: string | Date) => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
};

interface ChatMessageProps {
  message: ChatMessage;
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ message }) => {
  // Format timestamp for display
  const formattedTime = formatTime(message.timestamp);
  
  return (
    <div 
      className={cn(
        "flex mb-4 last:mb-0 gap-2",
        message.role === 'user' ? "justify-end" : "justify-start"
      )}
    >
      {message.role === 'assistant' && (
        <div className="flex items-start mt-1">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
            <Bot size={16} className="text-primary" />
          </div>
        </div>
      )}
      
      <div className={cn(
        "max-w-[80%] rounded-lg p-3",
        message.role === 'user' 
          ? "bg-primary text-primary-foreground rounded-tr-none" 
          : "bg-card-secondary text-card-foreground rounded-tl-none"
      )}>
        <div className="whitespace-pre-wrap break-words">
          {message.content}
        </div>
        <div className={cn(
          "text-xs mt-1",
          message.role === 'user' 
            ? "text-primary-foreground/70" 
            : "text-muted-foreground"
        )}>
          {formattedTime}
        </div>
      </div>
      
      {message.role === 'user' && (
        <div className="flex items-start mt-1">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
            <User size={16} className="text-primary" />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatMessageComponent;
