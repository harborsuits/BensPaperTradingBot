export type Quote = {
  symbol: string;
  price: number;
  open?: number;
  high?: number;
  low?: number;
  prevClose?: number;
  change?: number;           // vs prevClose
  changePct?: number;
  time: string;              // ISO
};

export interface QuoteProvider {
  getQuotes(symbols: string[]): Promise<Record<string, Quote>>;
}
