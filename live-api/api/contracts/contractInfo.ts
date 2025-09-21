/**
 * API Contract Versions
 *
 * Frontend can check these versions to ensure it's in sync with backend.
 * Bump version when schema changes to break old clients.
 */
export const CONTRACT_VERSIONS = {
  health: 'v1',
  decisions_recent: 'v1',
  brain_activity: 'v1',
  autoloop_status: 'v1',
  portfolio_summary: 'v1',
  paper_positions: 'v1',
  paper_orders: 'v1',
  metrics: 'v1',
} as const;

export type ContractVersions = typeof CONTRACT_VERSIONS;
