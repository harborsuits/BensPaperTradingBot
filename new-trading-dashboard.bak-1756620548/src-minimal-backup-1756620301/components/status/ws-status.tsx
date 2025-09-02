import { useEffect, useState } from 'react';

function getWsUrl(): string {
  const base = (import.meta.env.VITE_WS_URL as string) || 'ws://localhost:8000';
  return base.endsWith('/ws') ? base : `${base}/ws`;
}

export default function WsStatus() {
  const [state, setState] = useState('connecting');
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(getWsUrl());
      ws.onopen = () => setState('open');
      ws.onclose = () => setState('closed');
      ws.onerror = () => setState('error');
    } catch (e) {
      setState('error');
    }
    return () => ws?.close();
  }, []);
  return <div>WS: {state}</div>;
}


