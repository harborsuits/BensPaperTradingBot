import React, { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { X, Minimize2, Maximize2, Send, RefreshCw, Bot } from 'lucide-react';
import { aiChatApi, TradingContext } from '@/services/aiChatApi';
import useAIChatWebSocket, { ChatMessage } from '@/hooks/useAIChatWebSocket';
import ChatMessageComponent from './ChatMessage.tsx';

// Helper function to combine class names
const cn = (...classes: (string | undefined | null | false)[]) => {
  return classes.filter(Boolean).join(' ');
};

interface AIChatWidgetProps {
  contextData?: TradingContext;
}

const AIChatWidget: React.FC<AIChatWidgetProps> = ({ contextData }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [message, setMessage] = useState('');
  const [position, setPosition] = useState({ x: window.innerWidth - 380, y: window.innerHeight - 480 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const messageEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Connect to WebSocket
  useAIChatWebSocket();

  // Fetch chat history
  const { data: chatHistory, isLoading: isLoadingHistory } = useQuery(['ai', 'chat', 'history'], 
    () => aiChatApi.getHistory(), {
      refetchOnWindowFocus: false,
    }
  );

  // Send message mutation
  const { mutate: sendMessage, isLoading: isSending } = useMutation(
    (newMessage: string) => aiChatApi.sendMessage(newMessage, contextData),
    {
      onSuccess: () => {
        setMessage('');
      },
    }
  );

  // Clear history mutation
  const { mutate: clearHistory, isLoading: isClearing } = useMutation(
    () => aiChatApi.clearHistory(),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['ai', 'chat', 'history']);
      },
    }
  );

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  // Handle drag start
  const handleDragStart = (e: React.MouseEvent) => {
    if (!chatContainerRef.current) return;
    
    setIsDragging(true);
    const rect = chatContainerRef.current.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  // Handle dragging
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  // Handle message submit
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      sendMessage(message);
    }
  };

  // Toggle chat widget
  const toggleWidget = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setIsMinimized(false);
    }
  };

  // Toggle minimize/maximize
  const toggleMinimize = () => {
    setIsMinimized(!isMinimized);
  };

  useEffect(() => {
    if (chatContainerRef.current && isOpen) {
      chatContainerRef.current.style.top = `${position.y}px`;
      chatContainerRef.current.style.left = `${position.x}px`;
    }
  }, [position, isOpen]);

  return (
    <>
      {/* Floating button when chat is closed */}
      {!isOpen && (
        <button 
          onClick={toggleWidget}
          className="fixed z-50 flex items-center justify-center p-3 text-white transition-all duration-300 rounded-full shadow-lg bottom-4 right-4 bg-primary hover:bg-primary/90"
          aria-label="Open chat"
        >
          <Bot size={24} />
        </button>
      )}

      {/* Chat widget */}
      {isOpen && (
        <div 
          ref={chatContainerRef}
          className={cn(
            "fixed z-50 flex flex-col w-[350px] bg-card shadow-xl rounded-lg border border-border transition-all duration-300",
            isMinimized ? "h-[60px]" : "h-[480px]",
            isDragging ? 'cursor-grabbing' : 'cursor-auto'
          )}
        >
          {/* Chat header - used for dragging */}
          <div 
            className="flex items-center justify-between px-4 py-3 border-b cursor-move bg-card-secondary"
            onMouseDown={handleDragStart}
          >
            <div className="flex items-center gap-2">
              <Bot size={20} className="text-primary" />
              <h3 className="font-medium text-foreground">BenBot Assistant</h3>
            </div>
            <div className="flex items-center gap-2">
              <button 
                onClick={toggleMinimize} 
                className="p-1 text-muted-foreground hover:text-foreground"
                aria-label={isMinimized ? "Maximize" : "Minimize"}
              >
                {isMinimized ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
              </button>
              <button 
                onClick={toggleWidget} 
                className="p-1 text-muted-foreground hover:text-destructive"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Chat body - hidden when minimized */}
          {!isMinimized && (
            <>
              {/* Messages container */}
              <div className="flex-1 p-4 overflow-y-auto">
                {isLoadingHistory ? (
                  <div className="flex items-center justify-center h-full">
                    <RefreshCw size={24} className="text-muted-foreground animate-spin" />
                  </div>
                ) : chatHistory?.data?.messages && chatHistory.data.messages.length > 0 ? (
                  <>
                    {chatHistory.data.messages.map((msg: ChatMessage) => (
                      <ChatMessageComponent key={msg.id} message={msg} />
                    ))}
                    <div ref={messageEndRef} />
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground gap-2">
                    <Bot size={36} className="text-primary/40" />
                    <p>I'm your BenBot trading assistant. How can I help you today?</p>
                    <div className="grid grid-cols-2 w-full gap-2 mt-2">
                      <button 
                        onClick={() => sendMessage("What trading opportunities do you see today?")}
                        className="p-2 text-sm border rounded-md bg-card-secondary hover:bg-card-secondary/80"
                      >
                        Show opportunities
                      </button>
                      <button 
                        onClick={() => sendMessage("Analyze current market conditions")}
                        className="p-2 text-sm border rounded-md bg-card-secondary hover:bg-card-secondary/80"
                      >
                        Market analysis
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Input area */}
              <form 
                onSubmit={handleSubmit} 
                className="flex items-center gap-2 p-3 border-t"
              >
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Ask me anything about trading..."
                  className="flex-1 px-3 py-2 bg-background rounded-md focus:outline-none focus:ring-1 focus:ring-primary"
                  disabled={isSending}
                />
                <button
                  type="submit"
                  disabled={isSending || !message.trim()}
                  className="p-2 text-white rounded-md bg-primary hover:bg-primary/90 disabled:opacity-50"
                  aria-label="Send message"
                >
                  <Send size={18} />
                </button>
              </form>

              {/* Optional utilities footer */}
              <div className="flex justify-end px-3 py-1 text-xs border-t text-muted-foreground">
                <button
                  onClick={() => clearHistory()}
                  disabled={isClearing || (chatHistory?.data?.messages?.length || 0) === 0}
                  className="hover:text-foreground disabled:opacity-50"
                >
                  Clear history
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};

export default AIChatWidget;
