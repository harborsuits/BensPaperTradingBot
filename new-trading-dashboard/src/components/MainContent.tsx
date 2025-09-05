import React, { useState } from 'react'
import { PortfolioGraph } from './MainContent/PortfolioGraph'
import { AllocationBreakdown } from './MainContent/AllocationBreakdown'
import { RecentTrades } from './MainContent/RecentTrades'
import { TabNavigation, TabId } from './TabNavigation'
import { TabContent } from './TabContent'
import { Filters } from './Filters/Filters'
import { useAppDispatch, useAppSelector } from '../redux/hooks'
import {
  selectSelectedTickers,
  selectDateRange,
  selectSentimentFilter,
  setSelectedTickers,
  setDateRange,
  setSentiment,
} from '../redux/slices/globalFiltersSlice'

interface MainContentProps {}

const MainContent: React.FC<MainContentProps> = () => {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard')

  const selectedTickers = useAppSelector(selectSelectedTickers)
  const dateRange = useAppSelector(selectDateRange)
  const sentiment = useAppSelector(selectSentimentFilter)
  const dispatch = useAppDispatch()

  const handleTickersChange = (tickers: string[]) => {
    dispatch(setSelectedTickers(tickers))
  }

  const handleDateRangeChange = (range: { start: Date | null; end: Date | null }) => {
    dispatch(setDateRange(range))
  }

  const handleSentimentChange = (newSentiment: string) => {
    dispatch(setSentiment(newSentiment))
  }

  // Only show filters on tabs that need them
  const showFilters = ['dashboard', 'manual-backtesting', 'autonomous-backtesting', 'paper-trading', 'news-predictions'].includes(activeTab)

  return (
    <div className="main-content flex flex-col h-full overflow-hidden flex-1">
      {/* Header Bar */}
      <div className="header-bar flex items-center justify-between p-4 border-b border-border bg-card">
        {/* Logo */}
        <div className="logo-brand flex items-center space-x-2">
          <h1 className="text-2xl font-bold text-white">TradingBot</h1>
        </div>
        
        {/* Global Filters */}
        <div className="global-filters flex items-center space-x-4">
          {showFilters && (
            <Filters
              selectedTickers={selectedTickers}
              dateRange={dateRange}
              sentiment={sentiment}
              onTickersChange={handleTickersChange}
              onDateRangeChange={handleDateRangeChange}
              onSentimentChange={handleSentimentChange}
            />
          )}
        </div>
        
        {/* User menu */}
        <div className="user-menu flex items-center space-x-2">
          <div className="notifications relative">
            <button className="p-1 rounded-full hover:bg-muted/30">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"></path>
                <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"></path>
              </svg>
              <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
          </div>
          <button className="flex items-center space-x-2 p-1 rounded hover:bg-muted/30">
            <div className="avatar w-8 h-8 rounded-full bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-medium">BD</span>
            </div>
          </button>
        </div>
      </div>
      
      {/* Tab bar */}
      <div className="tab-bar p-4 pb-0">
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
      </div>
      
      {/* Tab Content in scrollable area */}
      <div className="flex-1 overflow-y-auto p-4">
        <TabContent 
          activeTab={activeTab} 
          selectedTickers={selectedTickers}
          dateRange={dateRange}
          sentiment={sentiment}
        />
      </div>
      
      {/* Floating AI Co-Pilot Button */}
      <div className="fixed bottom-4 right-4">
        <button className="w-12 h-12 bg-primary text-primary-foreground rounded-full flex items-center justify-center shadow-lg hover:bg-primary/90 transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 8V4H8"></path>
            <rect width="16" height="12" x="4" y="8" rx="2"></rect>
            <path d="M2 14h2"></path>
            <path d="M20 14h2"></path>
            <path d="M15 13v2"></path>
            <path d="M9 13v2"></path>
          </svg>
        </button>
      </div>
    </div>
  )
}

export default MainContent
