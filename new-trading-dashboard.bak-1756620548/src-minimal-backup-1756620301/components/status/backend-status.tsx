import { useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL ? String(import.meta.env.VITE_API_URL) : '/api';

export default function BackendStatus() {
  const [txt, setTxt] = useState('checking...');
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((j) => setTxt(`mode=${j?.flags?.TRADING_MODE ?? 'unknown'} paused=${j?.entries_paused ?? 'n/a'}`))
      .catch((e) => setTxt(`error: ${e}`));
  }, []);
  return <div>API: {txt}</div>;
}


