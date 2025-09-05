import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface DateRange {
  start: Date | null;
  end: Date | null;
}

interface GlobalFiltersState {
  selectedTickers: string[];
  dateRange: DateRange;
  sentiment: string;
}

const initialState: GlobalFiltersState = {
  selectedTickers: ['AAPL', 'MSFT', 'GOOGL'],
  dateRange: {
    start: null,
    end: null
  },
  sentiment: 'all'
};

export const globalFiltersSlice = createSlice({
  name: 'globalFilters',
  initialState,
  reducers: {
    setSelectedTickers: (state, action: PayloadAction<string[]>) => {
      state.selectedTickers = action.payload;
    },
    setDateRange: (state, action: PayloadAction<DateRange>) => {
      state.dateRange = action.payload;
    },
    setSentiment: (state, action: PayloadAction<string>) => {
      state.sentiment = action.payload;
    }
  }
});

export const { setSelectedTickers, setDateRange, setSentiment } = globalFiltersSlice.actions;

export const selectSelectedTickers = (state: RootState) => state.globalFilters.selectedTickers;
export const selectDateRange = (state: RootState) => state.globalFilters.dateRange;
export const selectSentimentFilter = (state: RootState) => state.globalFilters.sentiment;

export default globalFiltersSlice.reducer;
