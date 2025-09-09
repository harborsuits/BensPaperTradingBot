/**
 * Unified Data Refresh Orchestration System
 *
 * This service coordinates when different data sources should refresh based on:
 * - Data relationships and dependencies
 * - Refresh priorities (critical vs optional)
 * - System load and performance
 * - User activity patterns
 */

export type RefreshPriority = 'critical' | 'high' | 'medium' | 'low' | 'background';

export type DataSource = {
  name: string;
  queryKey: string[];
  endpoint: string;
  priority: RefreshPriority;
  baseInterval: number; // milliseconds
  dependencies: string[]; // names of other data sources this depends on
  dependents: string[]; // names of data sources that depend on this
  lastRefresh: number;
  refreshCount: number;
  errorCount: number;
};

export type RefreshSchedule = {
  dataSource: string;
  nextRefresh: number;
  priority: RefreshPriority;
  reason: string;
};

/**
 * Master registry of all data sources and their relationships
 */
const DATA_SOURCES: Record<string, DataSource> = {
  // Core trading data - highest priority
  marketContext: {
    name: 'marketContext',
    queryKey: ['marketContext'],
    endpoint: '/api/context',
    priority: 'critical',
    baseInterval: 30000, // 30 seconds
    dependencies: [],
    dependents: ['strategies', 'portfolio', 'decisions', 'pipeline'],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  strategies: {
    name: 'strategies',
    queryKey: ['strategies', 'active'],
    endpoint: '/api/strategies/active',
    priority: 'high',
    baseInterval: 60000, // 1 minute
    dependencies: ['marketContext'],
    dependents: ['portfolio', 'evoTester'],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  portfolio: {
    name: 'portfolio',
    queryKey: ['portfolio', 'paper'],
    endpoint: '/api/portfolio/paper',
    priority: 'high',
    baseInterval: 45000, // 45 seconds
    dependencies: ['strategies', 'marketContext'],
    dependents: ['decisions', 'trades'],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  // Real-time trading data
  decisions: {
    name: 'decisions',
    queryKey: ['decisions', 'recent'],
    endpoint: '/api/decisions/recent',
    priority: 'medium',
    baseInterval: 15000, // 15 seconds
    dependencies: ['portfolio', 'marketContext'],
    dependents: ['pipeline'],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  trades: {
    name: 'trades',
    queryKey: ['trades', 'recent'],
    endpoint: '/api/portfolio/paper/trades',
    priority: 'medium',
    baseInterval: 20000, // 20 seconds
    dependencies: ['portfolio'],
    dependents: ['pipeline'],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  // Evolution system data
  evoTester: {
    name: 'evoTester',
    queryKey: ['evoTester', 'history'],
    endpoint: '/api/evotester/history',
    priority: 'low',
    baseInterval: 300000, // 5 minutes
    dependencies: ['strategies'],
    dependents: [],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  evoSessions: {
    name: 'evoSessions',
    queryKey: ['evoTester', 'sessions'],
    endpoint: '/api/evotester/sessions/active',
    priority: 'high',
    baseInterval: 10000, // 10 seconds during active evolution
    dependencies: [],
    dependents: ['evoTester'],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  },

  // Pipeline monitoring
  pipeline: {
    name: 'pipeline',
    queryKey: ['pipeline', 'health'],
    endpoint: '/api/pipeline/health',
    priority: 'medium',
    baseInterval: 10000, // 10 seconds
    dependencies: ['decisions', 'trades', 'marketContext'],
    dependents: [],
    lastRefresh: 0,
    refreshCount: 0,
    errorCount: 0
  }
};

/**
 * Calculate dynamic refresh interval based on priority and system state
 */
const calculateDynamicInterval = (source: DataSource, systemLoad: number = 1): number => {
  const baseInterval = source.baseInterval;

  // Priority multipliers
  const priorityMultipliers = {
    critical: 0.5,   // 50% faster for critical data
    high: 0.75,      // 25% faster for high priority
    medium: 1.0,     // Base interval for medium
    low: 1.5,        // 50% slower for low priority
    background: 2.0  // 100% slower for background
  };

  // System load adjustment (higher load = slower refresh)
  const loadMultiplier = Math.max(0.5, Math.min(2.0, systemLoad));

  return Math.round(baseInterval * priorityMultipliers[source.priority] * loadMultiplier);
};

/**
 * Determine if a data source should refresh based on dependencies
 */
const shouldRefreshDueToDependency = (source: DataSource): boolean => {
  const now = Date.now();
  const timeSinceLastRefresh = now - source.lastRefresh;

  // Check if any dependencies were recently updated
  const dependenciesRecentlyUpdated = source.dependencies.some(depName => {
    const dep = DATA_SOURCES[depName];
    if (!dep) return false;
    const timeSinceDepUpdate = now - dep.lastRefresh;
    return timeSinceDepUpdate < 5000; // Within last 5 seconds
  });

  if (dependenciesRecentlyUpdated) {
    console.log(`[RefreshOrchestrator] ${source.name} refreshing due to dependency update`);
    return true;
  }

  // Check if it's time for regular refresh
  const dynamicInterval = calculateDynamicInterval(source);
  return timeSinceLastRefresh >= dynamicInterval;
};

/**
 * Generate optimized refresh schedule
 */
export const generateRefreshSchedule = (): RefreshSchedule[] => {
  const now = Date.now();
  const schedule: RefreshSchedule[] = [];

  Object.values(DATA_SOURCES).forEach(source => {
    if (shouldRefreshDueToDependency(source)) {
      const dynamicInterval = calculateDynamicInterval(source);
      const nextRefresh = now + dynamicInterval;

      schedule.push({
        dataSource: source.name,
        nextRefresh,
        priority: source.priority,
        reason: source.dependencies.length > 0 ? 'dependency_triggered' : 'scheduled'
      });
    }
  });

  // Sort by priority and time
  schedule.sort((a, b) => {
    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3, background: 4 };
    const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
    if (priorityDiff !== 0) return priorityDiff;
    return a.nextRefresh - b.nextRefresh;
  });

  return schedule;
};

/**
 * Execute refresh for a specific data source
 */
export const executeRefresh = async (dataSourceName: string, queryClient: any): Promise<boolean> => {
  const source = DATA_SOURCES[dataSourceName];
  if (!source) {
    console.error(`[RefreshOrchestrator] Unknown data source: ${dataSourceName}`);
    return false;
  }

  try {
    console.log(`[RefreshOrchestrator] Refreshing ${dataSourceName}`);

    // Invalidate the query to trigger refresh
    await queryClient.invalidateQueries({
      queryKey: source.queryKey,
      exact: true
    });

    // Update source statistics
    source.lastRefresh = Date.now();
    source.refreshCount++;

    // Also refresh dependents if this is a critical/high priority source
    if (source.priority === 'critical' || source.priority === 'high') {
      source.dependents.forEach(dependentName => {
        const dependent = DATA_SOURCES[dependentName];
        if (dependent) {
          setTimeout(() => {
            queryClient.invalidateQueries({
              queryKey: dependent.queryKey,
              exact: true
            });
          }, 1000); // Slight delay to prevent overwhelming the system
        }
      });
    }

    return true;
  } catch (error) {
    console.error(`[RefreshOrchestrator] Error refreshing ${dataSourceName}:`, error);
    source.errorCount++;
    return false;
  }
};

/**
 * Get system health metrics
 */
export const getSystemHealthMetrics = () => {
  const totalSources = Object.keys(DATA_SOURCES).length;
  const activeSources = Object.values(DATA_SOURCES).filter(s => s.refreshCount > 0).length;
  const errorRate = Object.values(DATA_SOURCES).reduce((sum, s) => sum + s.errorCount, 0) / totalSources;

  return {
    totalSources,
    activeSources,
    errorRate,
    averageRefreshInterval: Object.values(DATA_SOURCES).reduce((sum, s) => sum + calculateDynamicInterval(s), 0) / totalSources,
    lastRefreshTimes: Object.fromEntries(
      Object.entries(DATA_SOURCES).map(([name, source]) => [name, source.lastRefresh])
    )
  };
};

/**
 * Reset refresh statistics for monitoring
 */
export const resetRefreshStats = () => {
  Object.values(DATA_SOURCES).forEach(source => {
    source.refreshCount = 0;
    source.errorCount = 0;
  });
  console.log('[RefreshOrchestrator] Refresh statistics reset');
};
