import { useEffect, useState, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

/**
 * Cross-Component Dependency Management System
 *
 * This hook passively tracks data dependencies between components
 * and logs when dependencies are updated. Active refreshing is handled
 * by the DataRefreshOrchestrator service.
 */

export type Dependency = {
  queryKey: string[];
  description: string;
};

/**
 * Global dependency registry for cross-component data flow
 */
const dependencyRegistry: Record<string, Dependency[]> = {
  'EvolutionStatusBar': [
    { queryKey: ['marketContext'], description: 'EvolutionStatusBar depends on market context' },
    { queryKey: ['strategies', 'active'], description: 'EvolutionStatusBar depends on active strategies' },
    { queryKey: ['portfolio', 'paper'], description: 'EvolutionStatusBar depends on portfolio' },
    { queryKey: ['evoTester', 'history'], description: 'EvolutionStatusBar depends on evolution history' }
  ],
  'PipelineFlowVisualization': [
    { queryKey: ['pipeline', 'health'], description: 'PipelineFlowVisualization depends on pipeline health' },
    { queryKey: ['decisions', 'recent'], description: 'PipelineFlowVisualization depends on recent decisions' },
    { queryKey: ['trades', 'recent'], description: 'PipelineFlowVisualization depends on recent trades' },
    { queryKey: ['marketContext'], description: 'PipelineFlowVisualization depends on market context' }
  ],
  'StrategyDeploymentPipeline': [
    { queryKey: ['strategies', 'active'], description: 'StrategyDeploymentPipeline depends on active strategies' },
    { queryKey: ['evoTester', 'history'], description: 'StrategyDeploymentPipeline depends on evolution history' },
    { queryKey: ['portfolio', 'paper'], description: 'StrategyDeploymentPipeline depends on portfolio' }
  ],
  'EvoTesterDashboard': [
    { queryKey: ['evoTester', 'sessions'], description: 'EvoTesterDashboard depends on active sessions' }
  ]
};

/**
 * Hook for passive dependency tracking and logging
 */
export function useCrossComponentDependencies(componentName: string) {
  const queryClient = useQueryClient();
  const [lastDependencyCheck, setLastDependencyCheck] = useState<Date | null>(null);
  const dependencies = dependencyRegistry[componentName] || [];
  const activeListeners = useRef<(() => void)[]>([]);

  useEffect(() => {
    console.log(`[DependencyManager] Setting up ${dependencies.length} dependencies for ${componentName}`);

    activeListeners.current.forEach(unsubscribe => unsubscribe());
    activeListeners.current = [];

    dependencies.forEach(dep => {
      const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
        if (event.type === 'queryUpdated' && event.query.queryKey.every((key, i) => key === dep.queryKey[i])) {
          console.log(`[DependencyManager] ${componentName} detected update in ${dep.queryKey.join('/')}`);
          setLastDependencyCheck(new Date());
        }
      });
      activeListeners.current.push(unsubscribe);
    });

    return () => {
      activeListeners.current.forEach(unsubscribe => unsubscribe());
    };
  }, [componentName, dependencies, queryClient]);

  return { lastDependencyCheck, dependencies };
};

/**
 * Get dependency graph for visualization/debugging
 */
export const getDependencyGraph = () => {
  return dependencyRegistry;
};
