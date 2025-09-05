import { useEffect, useState } from 'react';
import { ScannerCandidate, ScannerResponse } from '@/types/market';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';

export function useScannerCandidates(pollMs = 30000) {
  const [items, setItems] = useState<ScannerCandidate[]>([]);
  const [asOf, setAsOf] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    let mounted = true;
    let t: any;
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/scanner/candidates`);
        const json = (await res.json()) as ScannerResponse | any;
        if (!mounted) return;
        const arr = Array.isArray(json?.items) ? json.items : [];
        setItems(arr);
        setAsOf(json?.asOf);
      } catch (e: any) {
        setError(e?.message || 'Failed to fetch candidates');
      }
      t = setTimeout(tick, pollMs);
    };
    tick();
    return () => { mounted = false; clearTimeout(t); };
  }, [pollMs]);

  return { items, asOf, error } as const;
}

import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export type Candidate = {
  symbol: string; last: number;
  score: number; confidence: number; side: "buy"|"sell";
  plan: { entry: number; stop: number; take: number; type: string };
  risk: { suggestedQty: number; spreadOK: boolean; liquidityOK: boolean };
  explain: { impact1h:number; impact24h:number; count24h:number; rvol:number; gapPct:number; spreadPct:number; atr:number; outlets:string[] };
  asOf: string;
};

export function useScannerCandidatesQuery(list = 'small_caps_liquid', limit = 30) {
  return useQuery<Candidate[]>({
    queryKey: ['scanner', list, limit],
    queryFn: async () => {
      const { data } = await axios.get(`/api/scanner/candidates`, { params: { list, limit }});
      const items = Array.isArray((data as any)?.items) ? (data as any).items : (Array.isArray(data) ? data : []);
      return items as Candidate[];
    },
    refetchInterval: 30_000, staleTime: 20_000, retry: 1
  });
}
