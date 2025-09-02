// This file provides mock data for development
export const mockAPI = {
  get: async (endpoint: string) => {
    console.log(`Mock API called with endpoint: ${endpoint}`);
    
    // Return data based on endpoint
    switch (endpoint) {
      case '/portfolio':
        return {
          data: {
            value: 105325.42,
            history: generatePortfolioHistory(),
            startDate: '2024-01-01',
            endDate: '2024-04-22',
            initialValue: 100000,
            currentValue: 105325.42,
            percentChange: 5.33,
          },
        };
      
      case '/allocation':
        return {
          data: {
            categories: [
              { name: 'Technology', value: 35 },
              { name: 'Finance', value: 25 },
              { name: 'Healthcare', value: 15 },
              { name: 'Consumer', value: 10 },
              { name: 'Energy', value: 5 },
              { name: 'Cash', value: 10 },
            ],
          },
        };
      
      case '/trades':
        return {
          data: {
            trades: [
              { id: 1, symbol: 'AAPL', date: '2024-04-20', type: 'BUY', quantity: 10, price: 169.42, total: 1694.20 },
              { id: 2, symbol: 'MSFT', date: '2024-04-19', type: 'SELL', quantity: 5, price: 402.75, total: 2013.75 },
              { id: 3, symbol: 'GOOGL', date: '2024-04-18', type: 'BUY', quantity: 8, price: 174.86, total: 1398.88 },
              { id: 4, symbol: 'AMZN', date: '2024-04-17', type: 'BUY', quantity: 12, price: 183.24, total: 2198.88 },
              { id: 5, symbol: 'TSLA', date: '2024-04-16', type: 'SELL', quantity: 6, price: 168.38, total: 1010.28 },
            ],
          },
        };
      
      case '/metrics':
        return {
          data: {
            metrics: [
              { name: 'Win Rate', value: 68.5, change: 2.3 },
              { name: 'Profit Factor', value: 2.14, change: 0.31 },
              { name: 'Sharpe Ratio', value: 1.72, change: -0.15 },
              { name: 'Max Drawdown', value: -5.2, change: 1.3 },
            ],
          },
        };
        
      case '/news':
        return {
          data: {
            items: [
              { id: 1, title: 'Fed Signals Potential Rate Cut in Next Meeting', source: 'Bloomberg', date: '2024-04-22', sentiment: 'positive' },
              { id: 2, title: 'AAPL Reports Strong Q1 Earnings', source: 'CNBC', date: '2024-04-21', sentiment: 'positive' },
              { id: 3, title: 'TSLA Production Issues in Berlin Factory', source: 'Reuters', date: '2024-04-20', sentiment: 'negative' },
              { id: 4, title: 'MSFT Cloud Services Growth Continues', source: 'WSJ', date: '2024-04-19', sentiment: 'positive' },
            ],
          },
        };
        
      default:
        return { data: {} };
    }
  },
  
  post: async (endpoint: string, data: any) => {
    console.log(`Mock API POST called with endpoint: ${endpoint}, data:`, data);
    return { data: { success: true, message: 'Operation successful' } };
  },
};

// Helper to generate portfolio history data
function generatePortfolioHistory() {
  const today = new Date();
  const history = [];
  let value = 100000;
  
  // Generate 90 days of data
  for (let i = 90; i >= 0; i--) {
    const date = new Date();
    date.setDate(today.getDate() - i);
    
    // Random daily change between -1.5% and +1.5%
    const dailyChange = (Math.random() * 3) - 1.5;
    value = value * (1 + (dailyChange / 100));
    
    history.push({
      date: date.toISOString().split('T')[0],
      value: Math.round(value * 100) / 100,
    });
  }
  
  return history;
} 