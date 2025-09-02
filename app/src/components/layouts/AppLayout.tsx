import React, { useState } from 'react';
import StatusBar from '@/components/ui/StatusBar';
import { useHeartbeat } from '@/hooks/useHeartbeat';
import { pingApi } from '@/services/health';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

interface AppLayoutProps {
  children: React.ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  // Market state - in production, get this from your API
  const [marketOpen, setMarketOpen] = useState(false);
  
  // Trading mode - in production, get this from your API/settings
  const [mode, setMode] = useState<'Paper' | 'Live'>('Paper');
  
  // Safety controls
  const [paused, setPaused] = useState(false);
  
  // Handle trading pause toggle
  const handlePauseToggle = () => {
    setPaused(!paused);
    // In production, call your API to actually pause trading
  };
  
  // Use heartbeat to monitor API health
  const heartbeat = useHeartbeat(pingApi, 5000);
  
  // Set up keyboard shortcuts
  useKeyboardShortcuts(handlePauseToggle);
  
  return (
    <div className="app-container">
      <StatusBar 
        mode={mode} 
        marketOpen={marketOpen} 
        heartbeat={heartbeat}
        paused={paused}
        onPauseToggle={handlePauseToggle}
      />
      
      <main className="main-content">
        {children}
      </main>
      
      <style jsx>{`
        .app-container {
          display: flex;
          flex-direction: column;
          min-height: 100vh;
          padding: 1rem;
          gap: 1rem;
        }
        
        .main-content {
          flex: 1;
        }
      `}</style>
    </div>
  );
}
