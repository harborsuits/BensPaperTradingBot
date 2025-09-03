import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { AccountSvc, PositionsSvc, OrdersSvc, StrategiesSvc, SignalsSvc, RiskSvc, HealthSvc, JobsSvc } from "@/services/trading";

export function useDashboardData() {
  const qc = useQueryClient();

  const account = useQuery({ queryKey:["account"], queryFn: AccountSvc.balance, refetchInterval: 5000 });
  const positions = useQuery({ queryKey:["positions"], queryFn: PositionsSvc.list, refetchInterval: 5000 });
  const ordersOpen = useQuery({ queryKey:["orders","open"], queryFn: OrdersSvc.open, refetchInterval: 5000 });
  const strategies = useQuery({ queryKey:["strategies"], queryFn: StrategiesSvc.list, refetchInterval: 5000 });
  const signals = useQuery({ queryKey:["signals","live"], queryFn: SignalsSvc.live, refetchInterval: 2000 });
  const risk = useQuery({ queryKey:["risk"], queryFn: RiskSvc.status, refetchInterval: 10000 });
  const health = useQuery({ queryKey:["health"], queryFn: HealthSvc.status, refetchInterval: 10000 });

  const placeOrder = useMutation({
    mutationFn: OrdersSvc.place,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey:["orders","open"] });
      qc.invalidateQueries({ queryKey:["positions"] });
    },
  });
  const cancelOrder = useMutation({
    mutationFn: OrdersSvc.cancel,
    onSuccess: () => qc.invalidateQueries({ queryKey:["orders","open"] }),
  });

  const startBacktest = useMutation({ mutationFn: JobsSvc.startBacktest });

  return { account, positions, ordersOpen, strategies, signals, risk, health, placeOrder, cancelOrder, startBacktest };
}
