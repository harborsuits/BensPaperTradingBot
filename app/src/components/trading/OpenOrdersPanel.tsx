import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Hourglass, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { formatDistanceToNowStrict } from 'date-fns';

const OpenOrdersPanel = () => {
  const queryClient = useQueryClient();
  const { data: openOrders, isLoading } = useQuery({
    queryKey: ['paper', 'orders', 'open'],
    queryFn: async () => {
      const res = await fetch('/api/paper/orders/open');
      if (!res.ok) return [];
      return await res.json();
    },
    refetchInterval: 5000,
  });

  const cancelOrderMutation = useMutation({
    mutationFn: async (orderId: string) => {
      const res = await fetch(`/api/paper/orders/${orderId}`, { method: 'DELETE' });
      if (!res.ok) {
        throw new Error(`Failed to cancel order: ${res.status} ${res.statusText}`);
      }
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper', 'orders', 'open'] });
    },
    onError: (error) => {
      console.error('Error cancelling order:', error);
    }
  });

  const getTimeRemaining = (createdAt) => {
    const createdDate = new Date(createdAt * 1000);
    const expiryDate = new Date(createdDate.getTime() + 5 * 60000); // 5 minutes expiry
    return formatDistanceToNowStrict(expiryDate, { addSuffix: true });
  };

  if (isLoading) {
    return <div className="text-sm text-muted-foreground">Loading open orders...</div>;
  }

  if (!openOrders || openOrders.length === 0) {
    return (
      <div className="text-center text-muted-foreground py-4">
        <Hourglass className="mx-auto h-6 w-6 mb-2 opacity-50" />
        <p className="text-sm">No open orders</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {openOrders.map(order => (
        <div key={order.order_id} className="flex items-center justify-between p-2 rounded-md bg-muted/50">
          <div className="flex items-center space-x-3 text-sm">
            <span className="font-bold">{order.symbol}</span>
            <span className="text-muted-foreground">•</span>
            <span>waiting to fill</span>
            <span className="text-muted-foreground">•</span>
            <span>expires in {getTimeRemaining(order.created_ts)}</span>
            <span className="text-muted-foreground">•</span>
            <span>Broker ID {order.order_id.slice(0, 3)}...{order.order_id.slice(-3)}</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => cancelOrderMutation.mutate(order.order_id)}
            disabled={cancelOrderMutation.isPending}
          >
            <Trash2 size={14} className="mr-1" />
            Cancel
          </Button>
        </div>
      ))}
    </div>
  );
};

export default OpenOrdersPanel;
