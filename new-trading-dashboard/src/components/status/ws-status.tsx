import { useEffect, useState } from 'react';
import { wsUrl } from '@/lib/wsUrl';

export default function WsStatus() {
  const [state, setState] = useState('connecting');
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl('/ws'));
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

