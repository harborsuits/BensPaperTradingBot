import React, { useEffect, useState } from 'react';
import { Bell, Check } from 'lucide-react';
import { alertsApi } from '@/services/alertsApi';

type AlertItem = {
  id: string;
  timestamp?: string;
  severity?: string;
  source?: string;
  message: string;
  acknowledged?: boolean;
};

interface Props {
  className?: string;
}

const NotificationBell: React.FC<Props> = ({ className = '' }) => {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    const r = await alertsApi.getRecent(10);
    if ((r as any).success && Array.isArray((r as any).data)) {
      setItems((r as any).data as AlertItem[]);
    } else {
      setItems([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);

  const unread = items.filter(i => !i.acknowledged).length;

  const ack = async (id: string) => {
    await alertsApi.acknowledge(id);
    setItems(prev => prev.map(i => i.id === id ? { ...i, acknowledged: true } : i));
  };

  return (
    <div className={`relative ${className}`}>
      <button className="p-2 rounded-full hover:bg-muted relative" onClick={() => setOpen(v => !v)}>
        <Bell size={20} />
        {unread > 0 && <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-primary"></span>}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-80 max-h-96 overflow-auto py-2 bg-card border border-border rounded-md shadow-lg z-50">
          <div className="px-3 pb-2 text-sm font-medium">Alerts</div>
          {loading && <div className="px-3 py-2 text-xs text-muted-foreground">Loading…</div>}
          {!loading && items.length === 0 && (
            <div className="px-3 py-2 text-xs text-muted-foreground">No alerts</div>
          )}
          {items.map((a) => (
            <div key={a.id} className="px-3 py-2 text-sm border-t border-border/50 flex items-start gap-2">
              <div className={`mt-1 h-2 w-2 rounded-full ${a.acknowledged ? 'bg-muted' : 'bg-red-500'}`} />
              <div className="flex-1">
                <div className="text-xs text-muted-foreground">
                  {(a.severity || 'info').toUpperCase()} {a.source ? `• ${a.source}` : ''} {a.timestamp ? `• ${new Date(a.timestamp).toLocaleTimeString()}` : ''}
                </div>
                <div>{a.message}</div>
              </div>
              {!a.acknowledged && (
                <button className="p-1 rounded hover:bg-muted" onClick={() => ack(a.id)} title="Acknowledge">
                  <Check size={16} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default NotificationBell;


