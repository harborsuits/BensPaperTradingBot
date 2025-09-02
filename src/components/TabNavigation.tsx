import React from 'react';

export type TabId = 'dashboard' | 'manual-backtesting' | 'autonomous-backtesting' | 'paper-trading' | 'news-predictions' | 'strategies' | 'live-trading';

interface TabNavigationProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
}

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'manual-backtesting', label: 'Manual Backtesting' },
    { id: 'autonomous-backtesting', label: 'Autonomous Backtesting' },
    { id: 'paper-trading', label: 'Paper Trading' },
    { id: 'news-predictions', label: 'News & Predictions' },
    { id: 'strategies', label: 'Strategies' },
    { id: 'live-trading', label: 'Live Trading' },
  ] as const;

  return (
    <div className="tab-navigation flex space-x-1 border-b border-border w-full">
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`px-6 py-3 text-sm font-medium transition-colors rounded-t-md
            ${activeTab === tab.id 
              ? 'bg-card border-b-2 border-primary text-primary' 
              : 'text-white hover:text-primary hover:bg-muted/30'
            }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
} 