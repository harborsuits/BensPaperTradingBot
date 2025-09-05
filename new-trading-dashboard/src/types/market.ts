export type Quote = {
  symbol: string;
  price: number;
  open?: number;
  high?: number;
  low?: number;
  prevClose?: number;
  change?: number;
  changePct?: number;
  time: string; // ISO
};

export type QuotesPayload = {
  asOf: string;
  quotes: Record<string, Quote>;
};

export type ScannerCandidate = {
  symbol: string;
  score?: number;
  notes?: string;
  reason?: string;
};

export type ScannerResponse = {
  items: ScannerCandidate[];
  asOf?: string;
};


