import { useState } from 'react';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

export default function RQDevtoolsToggle() {
  const initial = import.meta.env.DEV || import.meta.env.VITE_RQ_DEVTOOLS === 'true';
  const [open, setOpen] = useState<boolean>(Boolean(initial));
  return (
    <>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          position: 'fixed', right: 12, bottom: 12, zIndex: 9999,
          padding: '6px 10px', borderRadius: 8, border: '1px solid #ddd', background: '#fff',
          boxShadow: '0 4px 10px rgba(0,0,0,0.08)', fontSize: 12
        }}
      >
        {open ? 'Hide RQ' : 'Show RQ'}
      </button>
      <ReactQueryDevtools initialIsOpen={open} />
    </>
  );
}


