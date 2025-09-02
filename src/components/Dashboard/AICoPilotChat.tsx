import React, { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface AICoPilotChatProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
}

export function AICoPilotChat({ messages, onSendMessage }: AICoPilotChatProps) {
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newMessage.trim()) {
      onSendMessage(newMessage);
      setNewMessage('');
    }
  };

  return (
    <div className="ai-copilot-chat flex flex-col h-full">
      {/* Messages area */}
      <div className="messages-container flex-1 overflow-y-auto mb-3 space-y-3 max-h-[200px]">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message p-2 rounded-lg max-w-[90%] ${
              message.role === 'user' 
                ? 'bg-primary/10 ml-auto' 
                : 'bg-muted/30 mr-auto'
            }`}
          >
            <div className="text-sm text-white">{message.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Message input */}
      <form onSubmit={handleSubmit} className="flex items-center space-x-2">
        <input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Ask me anything about trading..."
          className="flex-1 p-2 bg-background border border-border rounded text-white"
        />
        <button 
          type="submit" 
          className="px-3 py-2 bg-primary text-primary-foreground rounded"
          disabled={!newMessage.trim()}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m22 2-7 20-4-9-9-4Z"></path>
            <path d="M22 2 11 13"></path>
          </svg>
        </button>
      </form>
    </div>
  );
} 