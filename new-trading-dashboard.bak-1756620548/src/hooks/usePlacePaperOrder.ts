import { useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";

type NewOrder = {
  symbol: string;
  side: "buy" | "sell";
  qty: number;
  type: "market" | "limit";
  price?: number;
};

async function pollOrder(id: string) {
  for (let i = 0; i < 12; i++) {
    const { data } = await axios.get(`/api/paper/orders/${id}`);
    if (["filled", "accepted", "rejected", "canceled"].includes(data.status)) return data;
    await new Promise(r => setTimeout(r, 1000));
  }
}

export function usePlacePaperOrder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (o: NewOrder) => {
      const { data } = await axios.post(`/api/paper/orders`, o);
      return (await pollOrder(data.id)) ?? data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["paperAccount"] });
      qc.invalidateQueries({ queryKey: ["paperPositions"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
      qc.invalidateQueries({ queryKey: ["portfolioHistory"] });
    },
  });
}


