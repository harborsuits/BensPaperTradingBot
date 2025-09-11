import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Decision, BrainHealth } from "@/schemas/trading";
import { useEffect, useState } from "react";

export function useDecisions(limit = 20) {
  return useQuery<Decision[]>({
    queryKey: ["decisions", limit],
    queryFn: async () => {
      const response = await fetch(`/api/decisions/recent?limit=${limit}`);
      const data = await response.json();
      return data.items || [];
    },
    refetchInterval: 5000, // Refetch every 5 seconds
    staleTime: 2000,
  });
}

export function useDecisionsStream() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const queryClient = useQueryClient();

  useEffect(() => {
    // Create WebSocket connection to backend server
    const ws = new WebSocket(`ws://localhost:4000/ws/decisions`);

    ws.onopen = () => {
      console.log('Connected to decisions stream');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'decision') {
          const newDecision = message.data;
          console.log('New decision received:', newDecision);

          // Update local state
          setDecisions(prev => [newDecision, ...prev.slice(0, 19)]); // Keep latest 20

          // Invalidate and refetch the decisions query
          queryClient.invalidateQueries({ queryKey: ["decisions"] });
        }
      } catch (error) {
        console.error('Failed to parse decision message:', error);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from decisions stream');
    };

    ws.onerror = (error) => {
      console.error('Decisions stream error:', error);
    };

    return () => {
      ws.close();
    };
  }, [queryClient]);

  return decisions;
}

export function useBrainHealth() {
  return useQuery<BrainHealth>({
    queryKey: ["brain", "health"],
    queryFn: async () => {
      const response = await fetch("/api/brain/health");
      return response.json();
    },
    refetchInterval: 30000, // Check every 30 seconds
    staleTime: 10000,
  });
}


