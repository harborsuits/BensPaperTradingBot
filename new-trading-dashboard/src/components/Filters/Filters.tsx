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
  onSentimentChange
}) => {
  return (
    <div className="filters flex items-center space-x-4">
      {/* Ticker selector */}
      <div>
        <select
          className="bg-background border border-border rounded p-1 text-white text-sm"
          value={selectedTickers[0] || ''}
          onChange={(e) => onTickersChange([e.target.value])}
        >
          <option value="AAPL">AAPL</option>
          <option value="MSFT">MSFT</option>
          <option value="GOOGL">GOOGL</option>
          <option value="AMZN">AMZN</option>
        </select>
      </div>
      
      {/* Sentiment filter */}
      <div>
        <select
          className="bg-background border border-border rounded p-1 text-white text-sm"
          value={sentiment}
          onChange={(e) => onSentimentChange(e.target.value)}
        >
          <option value="all">All Sentiment</option>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
        </select>
      </div>
    </div>
  );
};
