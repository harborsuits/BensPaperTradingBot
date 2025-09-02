import React from 'react';

interface FiltersProps {
  selectedTickers: string[];
  dateRange: { start: Date | null; end: Date | null };
  sentiment: string;
  onTickersChange: (tickers: string[]) => void;
  onDateRangeChange: (range: { start: Date | null; end: Date | null }) => void;
  onSentimentChange: (sentiment: string) => void;
}

export const Filters: React.FC<FiltersProps> = ({
  selectedTickers,
  dateRange,
  sentiment,
  onTickersChange,
  onDateRangeChange,
  onSentimentChange,
}) => {
  return (
    <div className="filters flex items-center space-x-4">
      {/* Ticker filter */}
      <div className="filter">
        <select 
          className="bg-background border border-border rounded p-1 text-sm"
          multiple
          value={selectedTickers}
          onChange={(e) => {
            const options = Array.from(e.target.selectedOptions, option => option.value);
            onTickersChange(options);
          }}
        >
          <option value="AAPL">AAPL</option>
          <option value="MSFT">MSFT</option>
          <option value="GOOGL">GOOGL</option>
          <option value="AMZN">AMZN</option>
        </select>
      </div>
      
      {/* Date Range filter */}
      <div className="filter flex items-center space-x-1">
        <input 
          type="date" 
          className="bg-background border border-border rounded p-1 text-sm"
          value={dateRange.start ? dateRange.start.toISOString().split('T')[0] : ''}
          onChange={(e) => onDateRangeChange({
            ...dateRange, 
            start: e.target.value ? new Date(e.target.value) : null
          })}
        />
        <span>-</span>
        <input 
          type="date" 
          className="bg-background border border-border rounded p-1 text-sm"
          value={dateRange.end ? dateRange.end.toISOString().split('T')[0] : ''}
          onChange={(e) => onDateRangeChange({
            ...dateRange,
            end: e.target.value ? new Date(e.target.value) : null
          })}
        />
      </div>
      
      {/* Sentiment filter */}
      <div className="filter">
        <select 
          className="bg-background border border-border rounded p-1 text-sm"
          value={sentiment}
          onChange={(e) => onSentimentChange(e.target.value)}
        >
          <option value="All">All Sentiment</option>
          <option value="Positive">Positive</option>
          <option value="Neutral">Neutral</option>
          <option value="Negative">Negative</option>
        </select>
      </div>
    </div>
  );
}; 