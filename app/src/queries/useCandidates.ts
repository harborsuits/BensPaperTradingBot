// src/queries/useCandidates.ts
import { useQuery, QueryFunctionContext } from "@tanstack/react-query";
import { normalizeCandidatesResponse, type Candidate } from "@/services/candidates";

async function fetchCandidates({ signal, queryKey }: QueryFunctionContext) {
  const [_, __, limit = 8] = queryKey as [string, string, number?];
  const ctrl = new AbortController();
  const link = signal ? new AbortController() : ctrl;
  const res = await fetch(`/api/scanner/candidates?limit=${limit}`, { signal: signal ?? ctrl.signal });
  
  // No retry on hard client errors
  if (!res.ok && res.status >= 400 && res.status < 500 && res.status !== 429) {
    const body = await res.text().catch(() => "");
    throw Object.assign(new Error(`Candidates ${res.status}`), { status: res.status, body });
  }
  
  const json = await res.json().catch(() => ({}));
  return normalizeCandidatesResponse(json);
}

export function useCandidates(limit: number = 8) {
  return useQuery<Candidate[]>({
    queryKey: ["candidates", "v1", limit],
    queryFn: fetchCandidates,
    placeholderData: [],
    staleTime: 15_000,
    gcTime: 5 * 60_000,
    refetchOnWindowFocus: "always",
    retry(failureCount, err: any) {
      const s = err?.status;
      if (s && s >= 400 && s < 500 && s !== 429) return false; // don't hammer 4xx (except 429)
      return failureCount < 2;
    },
  });
}

export function useTopCandidate() {
  const q = useCandidates(1);
  // Sort by score desc when present, then symbol asc
  const top = (q.data ?? [])
    .slice()
    .sort((a, b) => 
      ((b.score ?? -Infinity) - (a.score ?? -Infinity)) || 
      (a.symbol?.localeCompare(b.symbol || '') || 0)
    )[0];
  
  return { ...q, data: top };
}

// Helper to update candidates via WebSocket/SSE
export function updateCandidatesCache(queryClient: any, incoming: unknown) {
  queryClient.setQueryData<Candidate[]>(
    ["candidates", "v1"], 
    prev => normalizeCandidatesResponse(incoming) || prev || []
  );
}
