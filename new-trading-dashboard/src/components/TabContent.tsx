import React from 'react';
import { TabId } from './TabNavigation';
import DashboardPage from '../pages/DashboardPage';

interface TabContentProps {
  activeTab: TabId;
  selectedTickers: string[];
  dateRange: { start: Date | null; end: Date | null };
  sentiment: string;
}

export function TabContent({ activeTab, selectedTickers, dateRange, sentiment }: TabContentProps) {
  // Dashboard tab content
  if (activeTab === 'dashboard') {
    return <DashboardPage />;
  }
  
  // Manual Backtesting tab content
  if (activeTab === 'manual-backtesting') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4">Manual Backtesting</h2>
        <p className="text-muted-foreground">This tab is under construction.</p>
      </div>
    );
  }
  
  // Autonomous Backtesting tab content
  if (activeTab === 'autonomous-backtesting') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4">Autonomous Backtesting</h2>
        <p className="text-muted-foreground">This tab is under construction.</p>
      </div>
    );
  }
  
  // Paper Trading tab content
  if (activeTab === 'paper-trading') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4">Paper Trading</h2>
        <p className="text-muted-foreground">This tab is under construction.</p>
      </div>
    );
  }
  
  // News & Predictions tab content
  if (activeTab === 'news-predictions') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4">News & Predictions</h2>
        <p className="text-muted-foreground">This tab is under construction.</p>
      </div>
    );
  }
  
  // Strategies tab content
  if (activeTab === 'strategies') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4">Strategies</h2>
        <p className="text-muted-foreground">This tab is under construction.</p>
      </div>
    );
  }
  
  // Live Trading tab content
  if (activeTab === 'live-trading') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4">Live Trading</h2>
        <p className="text-muted-foreground">This tab is under construction.</p>
      </div>
    );
  }
  
  // Fallback
  return <div>Select a tab</div>;
}
