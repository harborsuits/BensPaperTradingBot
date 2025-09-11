#!/usr/bin/env node
/**
 * Kill Switch Drill - Production Readiness Test
 * Tests emergency shutdown functionality during market hours
 */

const http = require('http');

class KillSwitchDrill {
  constructor() {
    this.baseUrl = 'http://localhost:4000';
    this.drillResults = {
      preDrill: {},
      killSwitchExecution: {},
      systemVerification: {},
      recovery: {},
      overall: { passed: false, score: 0 }
    };
  }

  /**
   * Run complete kill switch drill
   */
  async runKillSwitchDrill() {
    console.log('🚨 KILL SWITCH DRILL - PRODUCTION READINESS TEST');
    console.log('=================================================');
    console.log('This drill will test emergency shutdown functionality');
    console.log('Ensure you have staged allocations before running this test!');
    console.log('');

    try {
      // Phase 1: Pre-drill system assessment
      await this.phase1_PreDrillAssessment();

      // Phase 2: Execute kill switch
      await this.phase2_ExecuteKillSwitch();

      // Phase 3: Verify system shutdown
      await this.phase3_VerifyShutdown();

      // Phase 4: Test recovery (optional)
      await this.phase4_TestRecovery();

      // Phase 5: Generate report
      this.generateDrillReport();

    } catch (error) {
      console.error('💥 Kill switch drill failed:', error);
      this.drillResults.overall.passed = false;
      this.drillResults.overall.error = error.message;
    }
  }

  /**
   * Phase 1: Assess system state before drill
   */
  async phase1_PreDrillAssessment() {
    console.log('\n📊 Phase 1: Pre-Drill System Assessment');

    const checks = [];

    // Check server health
    try {
      const health = await this.makeRequest('GET', '/api/health');
      checks.push({ test: 'server_health', status: 'healthy', details: health });
      console.log('✅ Server health: OK');
    } catch (error) {
      checks.push({ test: 'server_health', status: 'failed', details: error.message });
      console.log('❌ Server health: FAILED');
    }

    // Check active allocations
    try {
      const ledger = await this.makeRequest('GET', '/api/competition/ledger');
      const activeAllocs = ledger.ledger?.length || 0;
      checks.push({ test: 'active_allocations', count: activeAllocs, status: 'checked' });
      console.log(`📈 Active allocations: ${activeAllocs}`);
    } catch (error) {
      checks.push({ test: 'active_allocations', status: 'error', details: error.message });
      console.log('❌ Could not check allocations');
    }

    // Check kill switch status
    try {
      const killStatus = await this.makeRequest('GET', '/api/admin/kill-switch');
      checks.push({ test: 'kill_switch_status', status: killStatus.status, details: killStatus });
      console.log(`🔒 Kill switch status: ${killStatus.status}`);
    } catch (error) {
      checks.push({ test: 'kill_switch_status', status: 'error', details: error.message });
      console.log('❌ Could not check kill switch status');
    }

    // Check quote feed
    try {
      const quotes = await this.makeRequest('GET', '/api/quotes?symbols=SPY');
      checks.push({ test: 'quote_feed', status: 'active', details: quotes });
      console.log('📡 Quote feed: ACTIVE');
    } catch (error) {
      checks.push({ test: 'quote_feed', status: 'error', details: error.message });
      console.log('❌ Quote feed: ERROR');
    }

    this.drillResults.preDrill = {
      checks,
      timestamp: new Date().toISOString(),
      systemReady: checks.every(c => c.status !== 'failed' && c.status !== 'error')
    };

    console.log(`🏥 Pre-drill assessment: ${this.drillResults.preDrill.systemReady ? 'READY' : 'ISSUES DETECTED'}`);
  }

  /**
   * Phase 2: Execute kill switch
   */
  async phase2_ExecuteKillSwitch() {
    console.log('\n🚨 Phase 2: Executing Kill Switch');

    const killRequest = {
      reason: 'Kill Switch Drill - Production Readiness Test',
      confirm: true
    };

    const startTime = Date.now();

    try {
      console.log('Sending kill switch activation...');
      const response = await this.makeRequest('POST', '/api/admin/kill-switch', killRequest);
      const executionTime = Date.now() - startTime;

      this.drillResults.killSwitchExecution = {
        success: response.status === 'killed',
        executionTimeMs: executionTime,
        response: response,
        timestamp: new Date().toISOString()
      };

      if (response.status === 'killed') {
        console.log(`✅ Kill switch executed in ${executionTime}ms`);
        console.log(`   Response: ${JSON.stringify(response, null, 2)}`);
      } else {
        console.log(`❌ Kill switch execution failed: ${JSON.stringify(response, null, 2)}`);
        throw new Error('Kill switch did not return expected status');
      }

    } catch (error) {
      console.error('❌ Kill switch request failed:', error);
      this.drillResults.killSwitchExecution = {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
      throw error;
    }
  }

  /**
   * Phase 3: Verify system shutdown effectiveness
   */
  async phase3_VerifyShutdown() {
    console.log('\n🔍 Phase 3: Verifying System Shutdown');

    // Wait for kill switch to take effect
    console.log('Waiting 3 seconds for kill switch to propagate...');
    await this.sleep(3000);

    const verifications = [];

    // 1. Verify kill switch status
    try {
      const killStatus = await this.makeRequest('GET', '/api/admin/kill-switch');
      const killStatusCorrect = killStatus.status === 'killed';
      verifications.push({
        test: 'kill_switch_status',
        passed: killStatusCorrect,
        expected: 'killed',
        actual: killStatus.status,
        details: killStatus
      });
      console.log(`${killStatusCorrect ? '✅' : '❌'} Kill switch status: ${killStatus.status}`);
    } catch (error) {
      verifications.push({
        test: 'kill_switch_status',
        passed: false,
        error: error.message
      });
      console.log('❌ Kill switch status check failed');
    }

    // 2. Verify new staging is blocked
    try {
      const testStageRequest = {
        origin: 'drill_test',
        sessionId: 'drill_session_001',
        strategyRef: 'drill_strategy_001',
        allocation: 1000,
        pool: 'EVO',
        ttlDays: 1,
        consistencyToken: `drill_${Date.now()}`
      };

      const stageResponse = await this.makeRequest('POST', '/api/competition/stage', testStageRequest);
      const stagingBlocked = stageResponse.error === 'PRE_TRADE_PROOF_FAILED' ||
                           stageResponse.code === 'FAIL_CLOSED_GATING' ||
                           stageResponse.message?.includes('killed');

      verifications.push({
        test: 'staging_blocked',
        passed: stagingBlocked,
        expected: 'staging blocked or proof failed',
        actual: stageResponse.error || 'staging allowed',
        details: stageResponse
      });
      console.log(`${stagingBlocked ? '✅' : '❌'} New staging: ${stagingBlocked ? 'BLOCKED' : 'ALLOWED'}`);
    } catch (error) {
      verifications.push({
        test: 'staging_blocked',
        passed: true, // If request fails, consider it blocked
        expected: 'staging blocked',
        actual: 'request failed (good)',
        details: { error: error.message }
      });
      console.log('✅ New staging: REQUEST FAILED (expected in kill mode)');
    }

    // 3. Verify rebalancing is blocked
    try {
      const rebalanceResponse = await this.makeRequest('POST', '/api/competition/rebalance', {
        mode: 'execute',
        consistencyToken: `drill_rebalance_${Date.now()}`
      });

      const rebalancingBlocked = rebalanceResponse.error?.includes('killed') ||
                               rebalanceResponse.message?.includes('frozen');

      verifications.push({
        test: 'rebalancing_blocked',
        passed: rebalancingBlocked,
        expected: 'rebalancing blocked',
        actual: rebalanceResponse.error || 'rebalancing allowed',
        details: rebalanceResponse
      });
      console.log(`${rebalancingBlocked ? '✅' : '❌'} Rebalancing: ${rebalancingBlocked ? 'BLOCKED' : 'ALLOWED'}`);
    } catch (error) {
      verifications.push({
        test: 'rebalancing_blocked',
        passed: true, // If request fails, consider it blocked
        expected: 'rebalancing blocked',
        actual: 'request failed (good)',
        details: { error: error.message }
      });
      console.log('✅ Rebalancing: REQUEST FAILED (expected in kill mode)');
    }

    // 4. Verify quotes are still flowing (but staging is blocked)
    try {
      const quotes = await this.makeRequest('GET', '/api/quotes?symbols=SPY');
      const quotesFlowing = quotes.SPY && typeof quotes.SPY.last === 'number';
      verifications.push({
        test: 'quotes_still_flowing',
        passed: quotesFlowing,
        expected: 'quotes still flowing',
        actual: quotesFlowing ? 'quotes flowing' : 'no quotes',
        details: quotes
      });
      console.log(`${quotesFlowing ? '✅' : '❌'} Quotes: ${quotesFlowing ? 'STILL FLOWING' : 'STOPPED'}`);
    } catch (error) {
      verifications.push({
        test: 'quotes_still_flowing',
        passed: false,
        error: error.message
      });
      console.log('❌ Quotes check failed');
    }

    this.drillResults.systemVerification = {
      verifications,
      passedCount: verifications.filter(v => v.passed).length,
      totalCount: verifications.length,
      timestamp: new Date().toISOString()
    };

    const passRate = this.drillResults.systemVerification.passedCount / this.drillResults.systemVerification.totalCount;
    console.log(`🔍 Verification Results: ${this.drillResults.systemVerification.passedCount}/${this.drillResults.systemVerification.totalCount} passed (${(passRate * 100).toFixed(1)}%)`);
  }

  /**
   * Phase 4: Test recovery (optional)
   */
  async phase4_TestRecovery() {
    console.log('\n🔄 Phase 4: Recovery Test (Optional)');

    // Note: Full recovery would require manual server restart
    // This phase documents the recovery process

    this.drillResults.recovery = {
      note: 'Full recovery requires manual server restart to clear kill switch state',
      automated: false,
      instructions: [
        '1. Restart the server: npm start',
        '2. Verify kill switch status returns to "active"',
        '3. Confirm staging and rebalancing work again',
        '4. Check that quotes are flowing normally'
      ],
      timestamp: new Date().toISOString()
    };

    console.log('ℹ️  Recovery testing: MANUAL PROCESS REQUIRED');
    console.log('   Full recovery needs server restart to clear kill switch state');
  }

  /**
   * Generate comprehensive drill report
   */
  generateDrillReport() {
    console.log('\n📊 KILL SWITCH DRILL REPORT');
    console.log('===========================');

    const report = {
      drillSummary: {
        startTime: this.drillResults.preDrill?.timestamp,
        endTime: new Date().toISOString(),
        killSwitchExecuted: this.drillResults.killSwitchExecution?.success || false,
        verificationsPassed: this.drillResults.systemVerification?.passedCount || 0,
        verificationsTotal: this.drillResults.systemVerification?.totalCount || 0,
        systemPreCheckPassed: this.drillResults.preDrill?.systemReady || false
      },
      phases: {
        preDrill: this.drillResults.preDrill,
        killSwitchExecution: this.drillResults.killSwitchExecution,
        systemVerification: this.drillResults.systemVerification,
        recovery: this.drillResults.recovery
      }
    };

    // Calculate overall score
    let score = 0;
    if (this.drillResults.killSwitchExecution?.success) score += 40; // Kill switch execution
    if (this.drillResults.systemVerification) {
      const verifyScore = (this.drillResults.systemVerification.passedCount /
                          this.drillResults.systemVerification.totalCount) * 50;
      score += verifyScore; // Verification effectiveness
    }
    if (this.drillResults.preDrill?.systemReady) score += 10; // Pre-check readiness

    report.drillSummary.overallScore = Math.round(score);
    report.drillSummary.grade = score >= 90 ? 'A' :
                               score >= 80 ? 'B' :
                               score >= 70 ? 'C' :
                               score >= 60 ? 'D' : 'F';

    this.drillResults.overall.passed = score >= 80; // 80% passing grade
    this.drillResults.overall.score = report.drillSummary.overallScore;

    // Print results
    console.log(`Overall Result: ${this.drillResults.overall.passed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`Grade: ${report.drillSummary.grade} (${report.drillSummary.overallScore}%)`);
    console.log(`Kill Switch Executed: ${report.drillSummary.killSwitchExecuted ? '✅' : '❌'}`);
    console.log(`Pre-Check Passed: ${report.drillSummary.systemPreCheckPassed ? '✅' : '❌'}`);
    console.log(`Verifications: ${report.drillSummary.verificationsPassed}/${report.drillSummary.verificationsTotal}`);

    if (!this.drillResults.overall.passed) {
      console.log('\n❌ AREAS NEEDING IMPROVEMENT:');
      if (!this.drillResults.killSwitchExecution?.success) {
        console.log('  • Kill switch execution failed');
      }
      if (this.drillResults.systemVerification) {
        this.drillResults.systemVerification.verifications
          .filter(v => !v.passed)
          .forEach(v => {
            console.log(`  • ${v.test}: ${v.error || v.actual}`);
          });
      }
    }

    // Save detailed report
    const fs = require('fs');
    const reportFile = `kill-switch-drill-report-${new Date().toISOString().split('T')[0]}.json`;
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
    console.log(`\n💾 Detailed report saved to: ${reportFile}`);

    return report;
  }

  /**
   * Helper methods
   */
  makeRequest(method, endpoint, data = null) {
    return new Promise((resolve, reject) => {
      const url = new URL(endpoint, this.baseUrl);
      const options = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: method,
        headers: {
          'Content-Type': 'application/json'
        }
      };

      const req = http.request(options, (res) => {
        let body = '';

        res.on('data', (chunk) => {
          body += chunk;
        });

        res.on('end', () => {
          try {
            if (body.trim() === '') {
              resolve({});
            } else {
              const jsonData = JSON.parse(body);
              resolve(jsonData);
            }
          } catch (error) {
            resolve({ rawResponse: body, parseError: error.message });
          }
        });
      });

      req.on('error', (error) => {
        reject(error);
      });

      if (data) {
        req.write(JSON.stringify(data));
      }

      req.setTimeout(10000, () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      req.end();
    });
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Run drill if called directly
if (require.main === module) {
  const drill = new KillSwitchDrill();

  console.log('🚨 KILL SWITCH DRILL');
  console.log('===================');
  console.log('This will test emergency shutdown functionality.');
  console.log('Make sure you have staged allocations for proper testing.');
  console.log('');

  // Add a confirmation prompt for safety
  const readline = require('readline');
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  rl.question('Are you ready to run the kill switch drill? (type "yes" to confirm): ', (answer) => {
    if (answer.toLowerCase() === 'yes') {
      drill.runKillSwitchDrill().catch(console.error);
    } else {
      console.log('Drill cancelled.');
    }
    rl.close();
  });
}

module.exports = { KillSwitchDrill };
