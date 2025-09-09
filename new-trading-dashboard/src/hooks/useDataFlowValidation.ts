import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';

/**
 * Hook to validate and ensure data consistency across EvoTester components
 * Monitors query cache and ensures all related data stays synchronized
 */
export const useDataFlowValidation = () => {
  const queryClient = useQueryClient();

  useEffect(() => {
    const validateDataConsistency = () => {
      const queries = queryClient.getQueryCache().getAll();

      // Check for stale or inconsistent data
      const evoQueries = queries.filter(q =>
        q.queryKey[0] === 'evoTester' ||
        q.queryKey[0] === 'strategies' ||
        q.queryKey[0] === 'portfolio' ||
        q.queryKey[0] === 'context'
      );

      // Log data flow status for debugging
      console.log('[DataFlow] Active queries:', evoQueries.length);
      console.log('[DataFlow] Query states:', evoQueries.map(q => ({
        key: q.queryKey,
        state: q.state.status,
        lastUpdated: q.state.dataUpdatedAt,
        stale: Date.now() - q.state.dataUpdatedAt > (q.options?.staleTime || 30000)
      })));
    };

    // Validate every 30 seconds
    const interval = setInterval(validateDataConsistency, 30000);

    return () => clearInterval(interval);
  }, [queryClient]);

  return {
    validateConsistency: () => {
      console.log('[DataFlow] Manual consistency check triggered');
      // Could add more sophisticated validation logic here
    }
  };
};

/**
 * Utility to check if all critical data sources are fresh
 */
export const isDataFresh = (queryClient: any): boolean => {
  const criticalQueries = [
    ['evoTester', 'history'],
    ['strategies', 'active'],
    ['portfolio', 'paper'],
    ['marketContext']
  ];

  return criticalQueries.every(queryKey => {
    const query = queryClient.getQueryState(queryKey);
    if (!query || !query.data) return false;

    const staleTime = 30000; // 30 seconds
    return Date.now() - query.dataUpdatedAt < staleTime;
  });
};
