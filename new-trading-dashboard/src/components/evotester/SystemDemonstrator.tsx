import React, { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useCrossComponentDependencies, getDependencyGraph } from '../../hooks/useCrossComponentDependencies';
import { generateRefreshSchedule, executeRefresh, getSystemHealthMetrics } from '../../services/dataRefreshOrchestrator';

/**
 * System Demonstrator Component
 *
 * This component demonstrates the functionality of:
 * 1. Cross-Component Dependency Management
 * 2. Unified Data Refresh Orchestration
 */
export const SystemDemonstrator: React.FC = () => {
  const queryClient = useQueryClient();
  const { lastDependencyCheck, dependencies } = useCrossComponentDependencies('EvolutionStatusBar');
  const [refreshSchedule, setRefreshSchedule] = useState<any[]>([]);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [dependencyGraph, setDependencyGraph] = useState<any>(null);

  // Update refresh schedule every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      const schedule = generateRefreshSchedule();
      setRefreshSchedule(schedule);
      setSystemHealth(getSystemHealthMetrics());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Update dependency graph on mount
  useEffect(() => {
    setDependencyGraph(getDependencyGraph());
  }, []);

  const handleManualRefresh = async (dataSourceName: string) => {
    console.log(`[Demo] Manually triggering refresh for ${dataSourceName}`);
    const success = await executeRefresh(dataSourceName, queryClient);
    if (success) {
      console.log(`[Demo] Successfully refreshed ${dataSourceName}`);
    } else {
      console.error(`[Demo] Failed to refresh ${dataSourceName}`);
    }
  };

  return (
    <div className="system-demonstrator p-6 bg-gray-50 rounded-lg border">
      <h2 className="text-xl font-bold mb-4">System Functionality Demonstration</h2>

      {/* Cross-Component Dependency Management Demo */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">1. Cross-Component Dependency Management</h3>
        <div className="bg-card p-4 rounded border border-border text-foreground">
          <p className="text-sm text-foreground mb-2">
            Last dependency check: {lastDependencyCheck ? lastDependencyCheck.toLocaleTimeString() : 'None'}
          </p>
          <p className="text-sm mb-2">
            <strong>Dependencies for EvolutionStatusBar:</strong>
          </p>
          <ul className="text-sm list-disc list-inside space-y-1">
            {dependencies.map((dep, index) => (
              <li key={index}>
                {dep.queryKey.join('/')} - {dep.description}
              </li>
            ))}
          </ul>
          <p className="text-xs text-green-600 mt-2">
            ✅ System is actively monitoring these dependencies and logging updates to console
          </p>
        </div>
      </div>

      {/* Unified Data Refresh Orchestration Demo */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">2. Unified Data Refresh Orchestration</h3>
        <div className="bg-card p-4 rounded border border-border text-foreground">
          <div className="mb-4">
            <p className="text-sm mb-2">
              <strong>Current Refresh Schedule (next 5 items):</strong>
            </p>
            {refreshSchedule.slice(0, 5).map((item, index) => (
              <div key={index} className="text-xs p-2 bg-gray-100 rounded mb-1">
                {item.dataSource} ({item.priority}) - Next: {new Date(item.nextRefresh).toLocaleTimeString()} - {item.reason}
              </div>
            ))}
          </div>

          {systemHealth && (
            <div className="mb-4">
              <p className="text-sm mb-2">
                <strong>System Health Metrics:</strong>
              </p>
              <div className="text-xs space-y-1">
                <p>Total Sources: {systemHealth.totalSources}</p>
                <p>Active Sources: {systemHealth.activeSources}</p>
                <p>Error Rate: {(systemHealth.errorRate * 100).toFixed(1)}%</p>
                <p>Avg Refresh Interval: {(systemHealth.averageRefreshInterval / 1000).toFixed(1)}s</p>
              </div>
            </div>
          )}

          <div className="mb-4">
            <p className="text-sm mb-2">
              <strong>Manual Refresh Testing:</strong>
            </p>
            <div className="flex flex-wrap gap-2">
              {['marketContext', 'strategies', 'portfolio', 'decisions', 'evoTester'].map(dataSource => (
                <button
                  key={dataSource}
                  onClick={() => handleManualRefresh(dataSource)}
                  className="px-3 py-1 bg-primary text-primary-foreground text-xs rounded hover:bg-primary/90"
                >
                  Refresh {dataSource}
                </button>
              ))}
            </div>
          </div>

          <p className="text-xs text-green-600">
            ✅ System is actively scheduling and executing data refreshes based on priority and dependencies
          </p>
        </div>
      </div>

      {/* Dependency Graph Visualization */}
      {dependencyGraph && (
        <div>
          <h3 className="text-lg font-semibold mb-2">3. Dependency Graph Overview</h3>
          <div className="bg-card p-4 rounded border border-border text-foreground">
            <p className="text-sm mb-2">
              <strong>Components with Dependencies:</strong>
            </p>
            {Object.entries(dependencyGraph).map(([component, deps]: [string, any]) => (
              <div key={component} className="mb-3">
                <p className="text-sm font-medium">{component}</p>
                <ul className="text-xs list-disc list-inside ml-4 space-y-1">
                  {deps.map((dep: any, index: number) => (
                    <li key={index}>{dep.queryKey.join('/')}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 text-xs text-foreground">
        Check browser console for detailed logging of dependency updates and refresh operations.
      </div>
    </div>
  );
};
