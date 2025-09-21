#!/usr/bin/env node

/**
 * Type Generation Script
 *
 * Fetches API contracts from backend and generates TypeScript types.
 * This ensures frontend and backend stay in sync.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:4000';
const OUTPUT_DIR = path.join(__dirname, '..', 'src', 'types', 'generated');

async function fetchContracts() {
  try {
    console.log('📡 Fetching API contracts from backend...');
    const response = await fetch(`${BACKEND_URL}/api/contracts`);

    if (!response.ok) {
      throw new Error(`Failed to fetch contracts: ${response.status} ${response.statusText}`);
    }

    const contracts = await response.json();
    console.log('✅ Fetched contracts:', contracts);
    return contracts;
  } catch (error) {
    console.warn('⚠️  Could not fetch contracts from backend. Using fallback definitions.');
    console.warn('Make sure the backend is running on', BACKEND_URL);
    return {
      health: 'v1',
      decisions_recent: 'v1',
      brain_activity: 'v1',
      autoloop_status: 'v1',
      portfolio_summary: 'v1',
      paper_positions: 'v1',
      paper_orders: 'v1',
      metrics: 'v1',
    };
  }
}

function generateContractTypes(contracts) {
  const content = `/**
 * Generated API Contract Types
 *
 * Auto-generated from backend contracts endpoint.
 * Do not edit manually - run 'npm run gen:types' to regenerate.
 *
 * Generated at: ${new Date().toISOString()}
 */

export interface ContractVersions {
${Object.entries(contracts)
  .map(([key, version]) => `  ${key}: '${version}';`)
  .join('\n')}
}

// Contract version constants
export const CONTRACT_VERSIONS: ContractVersions = ${JSON.stringify(contracts, null, 2)};

// Type guards for contract validation
export function validateContractVersion(endpoint: keyof ContractVersions, version: string): boolean {
  return CONTRACT_VERSIONS[endpoint] === version;
}

export function getExpectedVersion(endpoint: keyof ContractVersions): string {
  return CONTRACT_VERSIONS[endpoint];
}
`;

  return content;
}

async function main() {
  try {
    // Ensure output directory exists
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    // Fetch contracts from backend
    const contracts = await fetchContracts();

    // Generate TypeScript content
    const typesContent = generateContractTypes(contracts);

    // Write to file
    const outputPath = path.join(OUTPUT_DIR, 'contracts.ts');
    fs.writeFileSync(outputPath, typesContent);

    console.log(`🎉 Generated types at: ${outputPath}`);
    console.log('📋 Contract versions:', contracts);

  } catch (error) {
    console.error('❌ Failed to generate types:', error.message);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { generateContractTypes, fetchContracts };
