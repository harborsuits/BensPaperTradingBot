/**
 * Crypto Provider Interface
 * Abstracts cryptocurrency exchange connectivity for unified trading
 */

export interface CryptoQuote {
  symbol: string;        // BTC/USD, ETH/USD, etc.
  price: number;         // Current price
  bid: number;          // Best bid price
  ask: number;          // Best ask price
  volume24h: number;    // 24h volume
  timestamp: string;    // ISO timestamp
  exchange: string;     // binance, coinbase, etc.
  spreadBps: number;    // Bid-ask spread in basis points
}

export interface CryptoOrderBook {
  symbol: string;
  bids: Array<[price: number, size: number]>;  // [price, size][]
  asks: Array<[price: number, size: number]>;  // [price, size][]
  timestamp: string;
  exchange: string;
}

export interface CryptoBalance {
  asset: string;        // BTC, ETH, USD, etc.
  free: number;         // Available balance
  locked: number;       // Locked in orders
  total: number;        // Total balance
}

export interface CryptoProvider {
  /**
   * Get current quotes for symbols
   */
  getQuotes(symbols: string[]): Promise<Record<string, CryptoQuote>>;

  /**
   * Get order book depth
   */
  getOrderBook(symbol: string, depth?: number): Promise<CryptoOrderBook>;

  /**
   * Get account balances
   */
  getBalances(): Promise<Record<string, CryptoBalance>>;

  /**
   * Check if provider is healthy
   */
  isHealthy(): Promise<boolean>;

  /**
   * Get exchange-specific fees
   */
  getFees(): Promise<Record<string, number>>;
}
