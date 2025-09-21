#!/usr/bin/env node

/**
 * Trigger and Audit Script - Tests coordination and risk gates
 * Waits for autopilot cycles and then audits the results
 */

const http = require('http');

const BASE_URL = 'http://localhost:4000';
const WAIT_TIME_MS = 90000; // 90 seconds for 3 autopilot cycles

function makeRequest(path) {
  return new Promise((resolve, reject) => {
    const url = `${BASE_URL}${path}`;
    http.get(url, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          resolve({ status: res.statusCode, data: jsonData });
        } catch (e) {
          resolve({ status: res.statusCode, data: data });
        }
      });
    }).on('error', reject);
  });
}

async function saveEvidence(name, data) {
  const fs = require('fs');
  const path = `evidence/${name}.json`;
  fs.writeFileSync(path, JSON.stringify(data, null, 2));
  console.log(`📄 Saved: ${path}`);
}

async function main() {
  console.log('🚀 Starting coordination and risk gate audit...\n');

  try {
    // 1. Capture initial state
    console.log('📊 Capturing initial state...');

    const initialStrategies = await makeRequest('/api/strategies');
    await saveEvidence('strategies_before', initialStrategies.data);

    const initialPositions = await makeRequest('/api/paper/positions');
    await saveEvidence('positions_before', initialPositions.data);

    // 2. Wait for autopilot cycles
    console.log(`⏳ Waiting ${WAIT_TIME_MS/1000} seconds for autopilot cycles...`);
    await new Promise(resolve => setTimeout(resolve, WAIT_TIME_MS));

    // 3. Capture coordination results
    console.log('🎯 Capturing coordination results...');

    const coordination = await makeRequest('/api/audit/coordination');
    await saveEvidence('coordination_results', coordination.data);

    // 4. Capture risk gate audit
    console.log('🛡️  Capturing risk gate audit...');

    const riskRejections = await makeRequest('/api/audit/risk-rejections');
    await saveEvidence('risk_rejections', riskRejections.data);

    // 5. Capture final state
    console.log('📈 Capturing final state...');

    const finalPositions = await makeRequest('/api/paper/positions');
    await saveEvidence('positions_after', finalPositions.data);

    const finalOrders = await makeRequest('/api/paper/orders');
    await saveEvidence('orders_after', finalOrders.data);

    // 6. Analyze results
    console.log('\n📋 ANALYSIS RESULTS:');
    console.log('='.repeat(50));

    // Coordination analysis
    if (coordination.status === 200 && coordination.data?.audit) {
      const audit = coordination.data.audit;
      console.log('🎯 COORDINATION RESULTS:');
      console.log(`  - Raw signals processed: ${audit.rawSignals || 0}`);
      console.log(`  - Signals scored: ${audit.scoredSignals || 0}`);
      console.log(`  - Winners selected: ${audit.winners || 0}`);
      console.log(`  - Signals rejected: ${audit.rejects || 0}`);
      console.log(`  - Conflicts resolved: ${audit.conflicts || 0}`);

      if (audit.conflicts > 0) {
        console.log('  ✅ Conflicts detected and resolved');
      } else {
        console.log('  ℹ️  No conflicts detected (may indicate single strategy dominance)');
      }
    }

    // Risk gate analysis
    if (riskRejections.status === 200 && riskRejections.data?.rejections) {
      const rejections = riskRejections.data.rejections;
      console.log('🛡️  RISK GATE RESULTS:');
      console.log(`  - Total rejections: ${rejections.length}`);

      if (rejections.length > 0) {
        // Group by reason
        const reasons = {};
        rejections.forEach(r => {
          reasons[r.reason] = (reasons[r.reason] || 0) + 1;
        });

        console.log('  - Rejection reasons:');
        Object.entries(reasons).forEach(([reason, count]) => {
          console.log(`    * ${reason}: ${count}`);
        });

        console.log('  ✅ Risk gates are active and rejecting unsafe trades');
      } else {
        console.log('  ℹ️  No trades rejected (may indicate all signals passed risk checks)');
      }
    }

    // Position changes
    const initialPosCount = Array.isArray(initialPositions.data) ? initialPositions.data.length : 0;
    const finalPosCount = Array.isArray(finalPositions.data) ? finalPositions.data.length : 0;
    const posChange = finalPosCount - initialPosCount;

    console.log('📊 POSITION CHANGES:');
    console.log(`  - Initial positions: ${initialPosCount}`);
    console.log(`  - Final positions: ${finalPosCount}`);
    console.log(`  - Net change: ${posChange > 0 ? '+' : ''}${posChange}`);

    if (posChange > 0) {
      console.log('  ✅ New positions opened by autopilot');
    } else if (posChange < 0) {
      console.log('  ✅ Positions closed by autopilot');
    } else {
      console.log('  ℹ️  No position changes (may indicate no signals or all rejected)');
    }

    console.log('\n✅ Audit complete! Evidence saved to evidence/ directory');

  } catch (error) {
    console.error('❌ Audit failed:', error.message);
    process.exit(1);
  }
}

// Ensure evidence directory exists
const fs = require('fs');
if (!fs.existsSync('evidence')) {
  fs.mkdirSync('evidence');
}

main();
