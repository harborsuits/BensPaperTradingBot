import React, { useState } from 'react';
import { PortfolioOverview } from './PortfolioOverview';
import { NewsColumn } from './NewsColumn';
import { AlertsPanel } from './AlertsPanel';
import { AICoPilotChat } from './AICoPilotChat';

export function Dashboard() {
  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([
    { role: 'assistant', content: 'How can I help you with your trading today?' }
  ]);
  
  const sendMessage = (message: string) => {
    // Add user message
    setChatMessages(prev => [...prev, { role: 'user', content: message }]);
    
    // Simulate AI response
    setTimeout(() => {
      setChatMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: `I've analyzed your request about "${message}". Based on current market conditions, I recommend considering momentum strategies for tech stocks. Would you like me to help you set up a backtest?` 
        }
      ]);
    }, 1000);
  };

  return (
    <div className="dashboard">
      {/* Portfolio Overview, Alerts Panel, and AI Co-Pilot Chat in a row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4 text-white">Portfolio Overview</h2>
          <PortfolioOverview />
        </div>
        
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4 text-white">Alerts Panel</h2>
          <AlertsPanel />
        </div>
        
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4 text-white">AI Co-Pilot Chat</h2>
          <AICoPilotChat messages={chatMessages} onSendMessage={sendMessage} />
        </div>
      </div>
      
      {/* News & Headlines section with 3 columns */}
      <div className="bg-card border border-border rounded-lg p-4 mb-4">
        <h2 className="text-lg font-semibold mb-4 text-white">News & Headlines</h2>
        <div className="grid grid-cols-3 gap-4">
          <NewsColumn category="sector" title="Sector" />
          <NewsColumn category="stock" title="Stock" />
          <NewsColumn category="macro" title="Macro" />
        </div>
      </div>
    </div>
  );
} 