import React, { useState } from 'react';
import { TabId } from './TabNavigation';
import { Dashboard } from './Dashboard/Dashboard';

interface TabContentProps {
  activeTab: TabId;
}

export function TabContent({ activeTab }: TabContentProps) {
  // Dashboard tab content
  if (activeTab === 'dashboard') {
    return <Dashboard />;
  }
  
  // Backtesting tab content - AUTONOMOUS, not manual
  if (activeTab === 'backtesting') {
    // State for automated runs
    const [isAutomatedRunning, setIsAutomatedRunning] = useState<boolean>(false);
    const [automationInterval, setAutomationInterval] = useState<number>(1); // minutes
    const [selectedDate, setSelectedDate] = useState<string>('2024-04-22');
    const [selectedTickers, setSelectedTickers] = useState<string[]>(['AAPL', 'MSFT', 'GOOGL']);
    const [configVersion, setConfigVersion] = useState<string>('v1.2');
    
    // Simulated backtest results
    const [results, setResults] = useState<any>({
      runSummary: "Scanned 20 tickers by sentiment+volatility, ran 3 strategies across 80 parameter combinations.",
      metrics: {
        sharpe: 1.85,
        winRate: 63,
        profitFactor: 2.1,
        maxDrawdown: -4.5
      },
      winners: [
        {
          id: 1,
          symbol: 'AAPL',
          strategy: 'Momentum',
          whyChosen: 'AAPL due to +0.85 sentiment & 30% vol spike',
          metrics: {
            sharpe: 2.12,
            winRate: 68.5,
            profitFactor: 2.05,
            maxDrawdown: -3.8
          },
          signals: ['MA50↗', 'RSI70↘'],
          summary: 'Avg +2.3% per day, 1 stop-out on vol spike',
          whatWorked: ['Tight stops on volatile days', 'Entry on RSI pullback'],
          whatDidnt: ['Too early exit on winning trades']
        },
        {
          id: 2,
          symbol: 'MSFT',
          strategy: 'Trend Following',
          whyChosen: 'MSFT due to +0.7 sentiment & strong earnings momentum',
          metrics: {
            sharpe: 1.95,
            winRate: 64.2,
            profitFactor: 1.89,
            maxDrawdown: -4.1
          },
          signals: ['EMA20↗', 'MACD+'],
          summary: 'Consistent gains with low drawdown, 2.1% average trade',
          whatWorked: ['Trend alignment with volume confirmation', 'Scaling in on pullbacks'],
          whatDidnt: ['Some false signals near market open']
        }
      ],
      insights: "Top 3 momentum picks show correlation of 0.45. Suggest portfolio diversification with non-correlated assets."
    });

    // Toggle automated runs
    const toggleAutomation = () => {
      setIsAutomatedRunning(!isAutomatedRunning);
    };

    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4 text-white">Backtesting (Autonomous)</h2>
        
        {/* Controls Bar */}
        <div className="controls-bar flex items-center justify-between bg-card border border-border rounded-lg p-4 mb-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <button 
                className={`px-4 py-2 rounded text-white ${isAutomatedRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}`}
                onClick={toggleAutomation}
              >
                {isAutomatedRunning ? 'Stop Automated Runs' : 'Run automated runs'}
              </button>
              <span className="mx-2 text-white">every</span>
              <input 
                type="number"
                min="1"
                max="60"
                value={automationInterval}
                onChange={(e) => setAutomationInterval(parseInt(e.target.value))}
                className="w-16 p-2 bg-background border border-border rounded text-white text-center"
              />
              <span className="ml-2 text-white">min</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <input 
                type="date" 
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="p-2 bg-background border border-border rounded text-white"
              />
              
              <select 
                className="p-2 bg-background border border-border rounded text-white"
                multiple
                value={selectedTickers}
                onChange={(e) => {
                  const options = Array.from(e.target.selectedOptions, option => option.value);
                  setSelectedTickers(options);
                }}
              >
                <option value="AAPL">AAPL</option>
                <option value="MSFT">MSFT</option>
                <option value="GOOGL">GOOGL</option>
                <option value="AMZN">AMZN</option>
                <option value="META">META</option>
              </select>
              
              <select 
                className="p-2 bg-background border border-border rounded text-white"
                value={configVersion}
                onChange={(e) => setConfigVersion(e.target.value)}
              >
                <option value="v1.0">Config v1.0</option>
                <option value="v1.1">Config v1.1</option>
                <option value="v1.2">Config v1.2</option>
              </select>
            </div>
          </div>
          
          <div>
            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
              isAutomatedRunning ? 'bg-green-500/20 text-green-500' : 'bg-yellow-500/20 text-yellow-500'
            }`}>
              {isAutomatedRunning ? 'Running' : 'Ready'}
            </span>
          </div>
        </div>
        
        {/* Run Summary */}
        <div className="run-summary bg-card border border-border rounded-lg p-4 mb-4">
          <p className="text-white">{results.runSummary}</p>
        </div>
        
        {/* KPI Strip */}
        <div className="kpi-strip grid grid-cols-4 gap-4 mb-6">
          <div className="kpi bg-card border border-border rounded-lg p-4 flex items-center">
            <div className="icon mr-3 text-primary">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m22 2-7 20-4-9-9-4Z"/>
                <path d="M22 2 11 13"/>
              </svg>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Sharpe</div>
              <div className="text-xl font-bold text-white">{results.metrics.sharpe}</div>
            </div>
          </div>
          
          <div className="kpi bg-card border border-border rounded-lg p-4 flex items-center">
            <div className="icon mr-3 text-primary">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="8"/>
                <path d="m15 9-3 3-3-3"/>
              </svg>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Win-Rate</div>
              <div className="text-xl font-bold text-white">{results.metrics.winRate}%</div>
            </div>
          </div>
          
          <div className="kpi bg-card border border-border rounded-lg p-4 flex items-center">
            <div className="icon mr-3 text-primary">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M2 20h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01M2 16h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01M2 12h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01M2 8h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01M2 4h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01m3.99 0h.01"/>
              </svg>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">PF</div>
              <div className="text-xl font-bold text-white">{results.metrics.profitFactor}</div>
            </div>
          </div>
          
          <div className="kpi bg-card border border-border rounded-lg p-4 flex items-center">
            <div className="icon mr-3 text-red-500">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z"/>
                <path d="M12 8v4"/>
                <path d="M12 16h.01"/>
              </svg>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">MaxDD</div>
              <div className="text-xl font-bold text-red-500">{results.metrics.maxDrawdown}%</div>
            </div>
          </div>
        </div>
        
        {/* Winners Table */}
        <div className="winners-table space-y-4 mb-6">
          <h3 className="text-lg font-semibold text-white">Winners (Top Results)</h3>
          
          {results.winners.map((winner: any) => (
            <div key={winner.id} className="winner bg-card border border-border rounded-lg overflow-hidden">
              {/* Why Chosen Banner */}
              <div className="why-chosen bg-primary/20 p-3 border-b border-border">
                <p className="text-primary font-medium">{winner.whyChosen}</p>
              </div>
              
              {/* Main Content */}
              <div className="p-4">
                {/* Symbol, Strategy and Metrics */}
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h4 className="text-lg font-semibold text-white">{winner.symbol} – {winner.strategy}</h4>
                    <div className="flex space-x-4 mt-1 text-sm">
                      <span className="text-muted-foreground">Sharpe <span className="text-white">{winner.metrics.sharpe}</span></span>
                      <span className="text-muted-foreground">WR <span className="text-white">{winner.metrics.winRate}%</span></span>
                      <span className="text-muted-foreground">PF <span className="text-white">{winner.metrics.profitFactor}</span></span>
                    </div>
                  </div>
                  
                  <div className="signals flex space-x-2">
                    {winner.signals.map((signal: string, index: number) => (
                      <span 
                        key={index}
                        className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium bg-primary/20 text-primary"
                      >
                        {signal}
                      </span>
                    ))}
                  </div>
                </div>
                
                {/* Summary */}
                <p className="text-white mb-3">{winner.summary}</p>
                
                {/* What Worked / What Didn't */}
                <div className="grid grid-cols-2 gap-4 mb-3">
                  <div>
                    <h5 className="text-sm font-medium text-green-500 mb-1">✅ What Worked</h5>
                    <ul className="text-sm text-white list-disc list-inside">
                      {winner.whatWorked.map((item: string, index: number) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h5 className="text-sm font-medium text-red-500 mb-1">❌ What Didn't</h5>
                    <ul className="text-sm text-white list-disc list-inside">
                      {winner.whatDidnt.map((item: string, index: number) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                
                {/* Controls */}
                <div className="flex justify-between items-center">
                  <div className="flex space-x-2">
                    <label className="flex items-center space-x-1 text-sm text-white">
                      <input type="checkbox" className="form-checkbox" />
                      <span>Show Trade Log</span>
                    </label>
                    
                    <label className="flex items-center space-x-1 text-sm text-white">
                      <input type="checkbox" className="form-checkbox" />
                      <span>Show Equity Curve</span>
                    </label>
                  </div>
                  
                  <button className="flex items-center space-x-1 px-3 py-1 bg-primary text-primary-foreground rounded text-sm">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="m9 18 6-6-6-6"/>
                    </svg>
                    <span>Promote to Paper Trading</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* Auto-Report Insights */}
        <div className="auto-report bg-card border border-border rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-2">Auto-Report Insights</h3>
          <p className="text-white">{results.insights}</p>
        </div>
      </div>
    );
  }
  
  // Paper Trading tab content
  if (activeTab === 'paper-trading') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4 text-white">Paper Trading</h2>
        
        <div className="bg-card border border-border rounded-lg p-6">
          {/* Account Selector */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1 text-white">Account</label>
            <select className="w-full p-2 bg-background border border-border rounded text-white">
              <option value="paper-1">Paper Account #1 ($25,000)</option>
              <option value="paper-2">Paper Account #2 ($100,000)</option>
            </select>
          </div>
          
          {/* Queued Strategies */}
          <h3 className="text-lg font-medium mb-3 text-white">Queued Strategies</h3>
          <div className="space-y-3 mb-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="strategy-card bg-muted/20 border border-border rounded-lg p-3">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-white">{['AAPL Momentum', 'MSFT Trend Following', 'ETF Basket'][i-1]}</h4>
                    <div className="text-sm text-muted-foreground mt-1">
                      <span className="text-green-500 font-medium">+2.3% P&L</span> • Promoted from backtest #{1000+i}
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button className="p-1 text-muted-foreground hover:text-white">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4Z"></path>
                      </svg>
                    </button>
                    <button className="p-1 text-muted-foreground hover:text-red-500">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 6h18"></path>
                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                        <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Deploy to Live Button */}
          <div className="flex justify-end">
            <button className="px-4 py-2 bg-green-600 text-white rounded flex items-center space-x-1 hover:bg-green-700">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 5v14"></path>
                <path d="m5 12 7 7 7-7"></path>
              </svg>
              <span>Deploy to Live Trading</span>
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  // News & Predictions tab content
  if (activeTab === 'news-predictions') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4 text-white">News & Predictions</h2>
        
        <div className="grid grid-cols-3 gap-4">
          {/* Column 1: Market News */}
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3 text-white">Market News</h3>
            <div className="space-y-4 overflow-y-auto max-h-[600px]">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="news-card border border-border rounded-lg overflow-hidden">
                  <div className="h-32 bg-background flex items-center justify-center">
                    <span className="text-muted-foreground">News Image</span>
                  </div>
                  <div className="p-3">
                    <h4 className="font-medium text-white">Market News Headline {i}</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Brief summary of the market news with quick take...
                    </p>
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-xs text-primary">2h ago • Reuters</span>
                      <span className="inline-flex items-center rounded-full px-2 py-1 text-xs bg-green-500/20 text-green-500">
                        Positive
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Column 2: Stock News */}
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3 text-white">Stock News</h3>
            <div className="space-y-4 overflow-y-auto max-h-[600px]">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="news-card border border-border rounded-lg overflow-hidden">
                  <div className="h-32 bg-background flex items-center justify-center">
                    <span className="text-muted-foreground">News Image</span>
                  </div>
                  <div className="p-3">
                    <h4 className="font-medium text-white">Stock News Headline {i}</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Brief summary of the stock news with quick take...
                    </p>
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-xs text-primary">4h ago • Bloomberg</span>
                      <span className="inline-flex items-center rounded-full px-2 py-1 text-xs bg-yellow-500/20 text-yellow-500">
                        Neutral
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Column 3: AI Predictions */}
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3 text-white">AI-Generated Advice</h3>
            <div className="space-y-4 overflow-y-auto max-h-[600px]">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="advice-card border border-border rounded-lg p-3">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-medium text-white">Trading Opportunity {i}</h4>
                    <span className="inline-flex items-center rounded-full px-2 py-1 text-xs bg-blue-500/20 text-blue-500">
                      AI Generated
                    </span>
                  </div>
                  <p className="text-sm text-white mb-2">
                    Based on recent news sentiment and technical patterns, consider a momentum play on AAPL with tight stops.
                  </p>
                  <div className="text-xs text-muted-foreground">
                    Generated from 5 recent articles • 87% confidence
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Strategies tab content
  if (activeTab === 'strategies') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4 text-white">Strategies</h2>
        
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="grid grid-cols-3 gap-6">
            {/* Strategy 1 */}
            <div className="strategy-config">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-white">Momentum</h3>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Fast Period</label>
                  <input type="range" min="5" max="50" defaultValue="10" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>5</span>
                    <span>50</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Slow Period</label>
                  <input type="range" min="20" max="200" defaultValue="50" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>20</span>
                    <span>200</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Signal Threshold</label>
                  <input type="range" min="0" max="100" defaultValue="70" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0</span>
                    <span>100</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Strategy 2 */}
            <div className="strategy-config">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-white">Mean Reversion</h3>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Lookback Period</label>
                  <input type="range" min="5" max="30" defaultValue="14" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>5</span>
                    <span>30</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Std Dev Threshold</label>
                  <input type="range" min="1" max="5" step="0.5" defaultValue="2" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>1</span>
                    <span>5</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Exit Period</label>
                  <input type="range" min="1" max="10" defaultValue="3" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>1</span>
                    <span>10</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Strategy 3 */}
            <div className="strategy-config">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-white">Trend Following</h3>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" />
                  <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">EMA Period</label>
                  <input type="range" min="10" max="100" defaultValue="20" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>10</span>
                    <span>100</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">ADX Min</label>
                  <input type="range" min="10" max="50" defaultValue="25" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>10</span>
                    <span>50</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-white">Trail Stop (%)</label>
                  <input type="range" min="1" max="10" step="0.5" defaultValue="2.5" className="w-full" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>1%</span>
                    <span>10%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 flex justify-end">
            <button className="px-4 py-2 bg-primary text-primary-foreground rounded">
              Save & Reload
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  // Live Trading tab content
  if (activeTab === 'live-trading') {
    return (
      <div className="tab-content">
        <h2 className="text-xl font-bold mb-4 text-white">Live Trading</h2>
        
        <div className="bg-card border border-border rounded-lg p-6">
          {/* Broker Status */}
          <div className="broker-status mb-6 flex items-center justify-between">
            <div className="flex items-center">
              <div className="status-indicator w-3 h-3 rounded-full bg-green-500 mr-2"></div>
              <span className="text-white">Broker Connected: Alpaca Securities</span>
            </div>
            
            <div className="flex space-x-4">
              <div className="text-white">
                <span className="text-muted-foreground">Account:</span> $125,450.67
              </div>
              <div className="text-green-500">
                <span className="text-muted-foreground">P&L Today:</span> +$1,245.30
              </div>
            </div>
          </div>
          
          {/* Real-time Blotter */}
          <div className="blotter mb-6">
            <h3 className="text-lg font-medium mb-3 text-white">Real-time Trading Blotter</h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 font-medium text-white">Time</th>
                    <th className="text-left py-2 px-3 font-medium text-white">Symbol</th>
                    <th className="text-left py-2 px-3 font-medium text-white">Action</th>
                    <th className="text-right py-2 px-3 font-medium text-white">Quantity</th>
                    <th className="text-right py-2 px-3 font-medium text-white">Price</th>
                    <th className="text-right py-2 px-3 font-medium text-white">Status</th>
                    <th className="text-right py-2 px-3 font-medium text-white">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    {time: '09:45:32', symbol: 'AAPL', action: 'BUY', qty: 100, price: 172.45, status: 'Filled', pnl: 125.30},
                    {time: '10:12:05', symbol: 'MSFT', action: 'SELL', qty: 50, price: 415.20, status: 'Filled', pnl: -78.50},
                    {time: '10:30:17', symbol: 'GOOGL', action: 'BUY', qty: 25, price: 172.50, status: 'Filled', pnl: 42.75},
                    {time: '11:05:44', symbol: 'META', action: 'BUY', qty: 30, price: 485.33, status: 'Filled', pnl: 156.25}
                  ].map((trade, index) => (
                    <tr key={index} className="border-b border-border">
                      <td className="py-2 px-3 text-white">{trade.time}</td>
                      <td className="py-2 px-3 text-white">{trade.symbol}</td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-0.5 text-xs rounded-full ${
                          trade.action === 'BUY' 
                            ? 'bg-green-500/20 text-green-500' 
                            : 'bg-red-500/20 text-red-500'
                        }`}>
                          {trade.action}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right text-white">{trade.qty}</td>
                      <td className="py-2 px-3 text-right text-white">${trade.price}</td>
                      <td className="py-2 px-3 text-right">
                        <span className="bg-green-500/20 text-green-500 px-2 py-0.5 text-xs rounded-full">
                          {trade.status}
                        </span>
                      </td>
                      <td className={`py-2 px-3 text-right ${trade.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {trade.pnl >= 0 ? '+' : ''}${Math.abs(trade.pnl).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Human vs Bot P&L Chart */}
          <div>
            <h3 className="text-lg font-medium mb-3 text-white">Human vs Bot P&L</h3>
            <div className="h-64 bg-muted/20 rounded flex items-center justify-center">
              <p className="text-muted-foreground">P&L comparison chart will appear here</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Fallback
  return <div className="text-white">Select a tab</div>;
} 