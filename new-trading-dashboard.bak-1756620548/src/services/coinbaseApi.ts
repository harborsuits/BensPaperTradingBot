/**
 * Coinbase API Client
 * 
 * This module provides functions to access the Coinbase market data from the backend.
 */

import axios from 'axios';
import { API_BASE_URL } from '../config';

// Types for Coinbase API responses
export interface TickerData {
  symbol: string;
  price: number;
  volume_24h: number;
  change_24h: number;
  high_24h: number;
  low_24h: number;
  timestamp: string;
}

export interface HistoricalCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderBookEntry {
  price: number;
  size: number;
}

export interface OrderBook {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  timestamp: string;
}

export interface MarketSummary {
  coins: {
    [symbol: string]: {
      price: number;
      volume_24h: number;
      change_24h_pct: number;
      last_updated: string;
    };
  };
  market_status: string;
  timestamp: string;
}

// API client functions
const coinbaseApi = {
  /**
   * Get available trading products from Coinbase
   */
  getProducts: async (): Promise<string[]> => {
    try {
      const response = await axios.get<string[]>(`${API_BASE_URL}/api/coinbase/products`);
      return response.data;
    } catch (error) {
      console.error('Error fetching Coinbase products:', error);
      return [];
    }
  },

  /**
   * Get ticker data for a specific symbol
   */
  getTicker: async (symbol: string): Promise<TickerData | null> => {
    try {
      const response = await axios.get<TickerData>(`${API_BASE_URL}/api/coinbase/ticker/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching ticker for ${symbol}:`, error);
      return null;
    }
  },

  /**
   * Get historical candles for a specific symbol
   */
  getCandles: async (
    symbol: string,
    timeframe: string = '1h',
    limit: number = 100
  ): Promise<HistoricalCandle[]> => {
    try {
      const response = await axios.get<HistoricalCandle[]>(
        `${API_BASE_URL}/api/coinbase/candles/${symbol}`,
        {
          params: { timeframe, limit }
        }
      );
      return response.data;
    } catch (error) {
      console.error(`Error fetching candles for ${symbol}:`, error);
      return [];
    }
  },

  /**
   * Get order book for a specific symbol
   */
  getOrderBook: async (symbol: string, level: number = 2): Promise<OrderBook | null> => {
    try {
      const response = await axios.get<OrderBook>(
        `${API_BASE_URL}/api/coinbase/orderbook/${symbol}`,
        {
          params: { level }
        }
      );
      return response.data;
    } catch (error) {
      console.error(`Error fetching order book for ${symbol}:`, error);
      return null;
    }
  },

  /**
   * Get a summary of the crypto market
   */
  getMarketSummary: async (): Promise<MarketSummary | null> => {
    try {
      const response = await axios.get<MarketSummary>(`${API_BASE_URL}/api/coinbase/market-summary`);
      return response.data;
    } catch (error) {
      console.error('Error fetching market summary:', error);
      return null;
    }
  },
};

export default coinbaseApi;
