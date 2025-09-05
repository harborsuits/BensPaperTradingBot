import { create } from 'zustand';
import { pricesStream, decisionsStream } from '@/lib/streams';

type LiveStatus = {
  prices: { state: string; staleMs: number };
  decisions: { state: string; staleMs: number };
  setTick: () => void;
};

export const useLiveStatus = create<LiveStatus>((set) => {
  // periodic sampler
  setInterval(() => set({}), 1000);

  return {
    prices: { state: pricesStream.getState(), staleMs: pricesStream.getLastMessageAgeMs() },
    decisions: { state: decisionsStream.getState(), staleMs: decisionsStream.getLastMessageAgeMs() },
    setTick: () => set({
      prices: { state: pricesStream.getState(), staleMs: pricesStream.getLastMessageAgeMs() },
      decisions: { state: decisionsStream.getState(), staleMs: decisionsStream.getLastMessageAgeMs() },
    })
  };
});
