import React from 'react';
import { useQuery } from '@tanstack/react-query';

type OrderRole = 'entry' | 'stop' | 'target';

interface Order {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  limit_price?: number;
  stop_price?: number;
  status: 'accepted' | 'replaced' | 'pending_cancel' | 'filled' | 'canceled' | 'rejected';
  created_ts: number;
  oco_group_id?: string;
  role?: OrderRole;
  tif_seconds?: number;
  attach?: Record<string, any>;
}

function formatPrice(price?: number) {
  if (price == null) return '—';
  return price.toFixed(2);
}

function formatTimeInForce(created: number, tif?: number) {
  if (!tif) return '—';
  
  const now = Date.now() / 1000;
  const expiresAt = created + tif;
  const secondsLeft = Math.max(0, Math.floor(expiresAt - now));
  
  if (secondsLeft <= 0) return 'Expired';
  
  const minutes = Math.floor(secondsLeft / 60);
  const seconds = secondsLeft % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function getRoleBadgeClass(role?: OrderRole) {
  switch (role) {
    case 'entry': return 'bg-blue-900/50 border-blue-700 text-blue-300';
    case 'stop': return 'bg-rose-900/50 border-rose-700 text-rose-300';
    case 'target': return 'bg-emerald-900/50 border-emerald-700 text-emerald-300';
    default: return 'bg-neutral-800 border-neutral-700 text-neutral-300';
  }
}

export default function OpenOrdersPanel() {
  const { data: orders = [], isLoading, error, refetch } = useQuery<Order[]>({
    queryKey: ['open-orders'],
    queryFn: async () => {
      const res = await fetch('/api/paper/orders/open');
      if (!res.ok) throw new Error('Failed to fetch open orders');
      return res.json();
    },
    refetchInterval: 3000,
  });

  const cancelOrder = async (orderId: string) => {
    try {
      const res = await fetch(`/api/paper/orders/${orderId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'user_requested' }),
      });
      
      if (!res.ok) throw new Error('Failed to cancel order');
      refetch();
    } catch (err) {
      console.error('Error canceling order:', err);
    }
  };

  if (isLoading) {
    return <div className="p-4 text-center text-neutral-400">Loading orders...</div>;
  }

  if (error) {
    return <div className="p-4 text-center text-rose-400">Error loading orders</div>;
  }

  if (!orders.length) {
    return <div className="p-4 text-center text-neutral-400">No open orders</div>;
  }

  // Group orders by OCO group
  const orderGroups: Record<string, Order[]> = {};
  
  orders.forEach(order => {
    const groupId = order.oco_group_id || `single-${order.order_id}`;
    if (!orderGroups[groupId]) {
      orderGroups[groupId] = [];
    }
    orderGroups[groupId].push(order);
  });

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="sticky top-0 bg-black/50 backdrop-blur-sm">
          <tr className="border-b border-neutral-800">
            <th className="text-left py-2 px-2">Symbol</th>
            <th className="text-left py-2 px-2">Side</th>
            <th className="text-left py-2 px-2">Qty</th>
            <th className="text-left py-2 px-2">Price</th>
            <th className="text-left py-2 px-2 hidden sm:table-cell">Role</th>
            <th className="text-left py-2 px-2 hidden md:table-cell">TIF</th>
            <th className="text-right py-2 px-2">Action</th>
          </tr>
        </thead>
        <tbody>
          {Object.values(orderGroups).map((group, groupIndex) => (
            <React.Fragment key={`group-${groupIndex}`}>
              {group.map((order, i) => (
                <tr 
                  key={order.order_id} 
                  className={`
                    ${i < group.length - 1 ? 'border-b border-dashed border-neutral-800/50' : 'border-b border-neutral-800'}
                    ${i === 0 && groupIndex > 0 ? 'border-t-4 border-t-neutral-800' : ''}
                  `}
                >
                  <td className="py-2 px-2 font-medium">{order.symbol}</td>
                  <td className="py-2 px-2">
                    <span className={`px-2 py-0.5 rounded text-xs ${order.side === 'BUY' ? 'bg-emerald-900/30 text-emerald-300' : 'bg-rose-900/30 text-rose-300'}`}>
                      {order.side}
                    </span>
                  </td>
                  <td className="py-2 px-2 tabular-nums">{order.qty}</td>
                  <td className="py-2 px-2 tabular-nums">
                    {order.limit_price ? formatPrice(order.limit_price) : order.stop_price ? `${formatPrice(order.stop_price)} (stop)` : '—'}
                  </td>
                  <td className="py-2 px-2 hidden sm:table-cell">
                    <span className={`px-2 py-0.5 rounded text-xs border ${getRoleBadgeClass(order.role)}`}>
                      {order.role || 'unknown'}
                    </span>
                  </td>
                  <td className="py-2 px-2 hidden md:table-cell tabular-nums">
                    {formatTimeInForce(order.created_ts, order.tif_seconds)}
                  </td>
                  <td className="py-2 px-2 text-right">
                    <button 
                      onClick={() => cancelOrder(order.order_id)}
                      className="px-2 py-1 text-xs rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200"
                    >
                      Cancel
                    </button>
                  </td>
                </tr>
              ))}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}