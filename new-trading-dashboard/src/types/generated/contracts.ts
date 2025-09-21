/**
 * Generated API Contract Types
 *
 * Auto-generated from backend contracts endpoint.
 * Do not edit manually - run 'npm run gen:types' to regenerate.
 *
 * Generated at: 2025-09-15T05:10:46.728Z
 */

export interface ContractVersions {
  health: 'v1';
  decisions_recent: 'v1';
  brain_activity: 'v1';
  autoloop_status: 'v1';
  portfolio_summary: 'v1';
  paper_positions: 'v1';
  paper_orders: 'v1';
  metrics: 'v1';
}

// Contract version constants
export const CONTRACT_VERSIONS: ContractVersions = {
  "health": "v1",
  "decisions_recent": "v1",
  "brain_activity": "v1",
  "autoloop_status": "v1",
  "portfolio_summary": "v1",
  "paper_positions": "v1",
  "paper_orders": "v1",
  "metrics": "v1"
};

// Type guards for contract validation
export function validateContractVersion(endpoint: keyof ContractVersions, version: string): boolean {
  return CONTRACT_VERSIONS[endpoint] === version;
}

export function getExpectedVersion(endpoint: keyof ContractVersions): string {
  return CONTRACT_VERSIONS[endpoint];
}
