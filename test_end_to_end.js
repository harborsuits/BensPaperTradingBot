#!/usr/bin/env node

/**
 * END-TO-END SAFETY PROOF TESTING
 * Tests complete integration between backend and frontend
 */

const axios = require('axios');

const BACKEND_URL = 'http://localhost:4000';
const FRONTEND_URL = 'http://localhost:3003';

class EndToEndTester {
  constructor() {
    this.results = { tests: [], summary: { total: 0, passed: 0, failed: 0 } };
    this.startTime = new Date();
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = {
      'info': 'ℹ️ ',
      'success': '✅',
      'error': '❌',
      'warning': '⚠️ '
    }[type] || '';

    console.log(`[${timestamp}] ${prefix} ${message}`);
  }

  addTest(name, result, details = {}) {
    this.results.tests.push({ name, result, details, timestamp: new Date().toISOString() });
    this.results.summary.total++;

    if (result) {
      this.results.summary.passed++;
      this.log(`${name} - PASSED`, 'success');
    } else {
      this.results.summary.failed++;
      this.log(`${name} - FAILED`, 'error');
    }

    if (details.message) {
      console.log(`   ${details.message}`);
    }
  }

  async testBackendHealth() {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/health`);
      const isHealthy = response.status === 200;
      this.addTest('Backend Health Check', isHealthy, {
        status: response.status,
        message: isHealthy ? 'Backend is responding' : 'Backend health check failed'
      });
      return isHealthy;
    } catch (error) {
      this.addTest('Backend Health Check', false, {
        error: error.message,
        message: 'Backend is not responding'
      });
      return false;
    }
  }

  async testSafetyProofEndpoints() {
    const endpoints = [
      '/api/proofs/summary?window=24h',
      '/api/proofs/fills?since=-24h',
      '/api/proofs/idempotency',
      '/api/proofs/attestation/latest'
    ];

    for (const endpoint of endpoints) {
      try {
        const response = await axios.get(`${BACKEND_URL}${endpoint}`);
        const isValid = response.status === 200 && response.data;
        this.addTest(`Safety Proof Endpoint: ${endpoint}`, isValid, {
          status: response.status,
          hasData: !!response.data,
          message: isValid ? 'Endpoint responding with data' : 'Endpoint not working'
        });
      } catch (error) {
        this.addTest(`Safety Proof Endpoint: ${endpoint}`, false, {
          error: error.message,
          message: 'Endpoint not accessible'
        });
      }
    }
  }

  async testOptionsTradingAPI() {
    // Test position sizing with proof mode
    try {
      const testData = {
        symbol: 'SPY',
        optionType: 'long_call',
        capital: 1000,
        underlyingPrice: 450,
        strike: 455,
        expiry: '2025-12-31',
        ivRank: 0.5,
        expectedMove: 0.03,
        conviction: 0.1,
        chainQuality: {
          overall: 0.8,
          spreadScore: 0.85,
          volumeScore: 0.75,
          oiScore: 0.7
        },
        frictionBudget: 0.20,
        proof: true
      };

      const response = await axios.post(`${BACKEND_URL}/api/options/position-size`, testData);
      const hasProof = response.data && response.data.proof;
      const hasSizing = response.data && response.data.contracts !== undefined;

      this.addTest('Options Position Sizing API', hasSizing && hasProof, {
        hasSizing: hasSizing,
        hasProof: hasProof,
        contracts: response.data?.contracts,
        message: hasSizing && hasProof ? 'Position sizing with proofs working' : 'Position sizing or proofs missing'
      });
    } catch (error) {
      this.addTest('Options Position Sizing API', false, {
        error: error.message,
        message: 'Position sizing API not working'
      });
    }

    // Test route selection
    try {
      const testData = {
        symbol: 'SPY',
        capital: 1000,
        proof: true
      };

      const response = await axios.post(`${BACKEND_URL}/api/options/route-selection`, testData);
      const hasSelection = response.data && response.data.selection;
      const hasProof = response.data && response.data.proof;

      this.addTest('Options Route Selection API', hasSelection && hasProof, {
        hasSelection: hasSelection,
        hasProof: hasProof,
        selectedRoute: response.data?.selection?.route,
        message: hasSelection && hasProof ? 'Route selection with proofs working' : 'Route selection or proofs missing'
      });
    } catch (error) {
      this.addTest('Options Route Selection API', false, {
        error: error.message,
        message: 'Route selection API not working'
      });
    }
  }

  async testFrontendHealth() {
    try {
      const response = await axios.get(FRONTEND_URL);
      const isHealthy = response.status === 200;
      this.addTest('Frontend Health Check', isHealthy, {
        status: response.status,
        message: isHealthy ? 'Frontend is serving pages' : 'Frontend not responding'
      });
      return isHealthy;
    } catch (error) {
      this.addTest('Frontend Health Check', false, {
        error: error.message,
        message: 'Frontend is not accessible'
      });
      return false;
    }
  }

  async testSafetyProofIntegration() {
    // Test that frontend can fetch safety proofs from backend
    try {
      // Simulate what the frontend SafetyStatusChip component does
      const summaryResponse = await axios.get(`${BACKEND_URL}/api/proofs/summary?window=24h`);
      const attestationResponse = await axios.get(`${BACKEND_URL}/api/proofs/attestation/latest`);

      const hasSummary = summaryResponse.data && summaryResponse.data.nbboFreshPct !== undefined;
      const hasAttestation = attestationResponse.data && attestationResponse.data.commit;

      this.addTest('Safety Proof Frontend Integration', hasSummary && hasAttestation, {
        hasSummary: hasSummary,
        hasAttestation: hasAttestation,
        nbboFreshPct: summaryResponse.data?.nbboFreshPct,
        commit: attestationResponse.data?.commit?.substring(0, 7),
        message: hasSummary && hasAttestation ? 'Frontend can fetch safety proofs' : 'Safety proof integration incomplete'
      });
    } catch (error) {
      this.addTest('Safety Proof Frontend Integration', false, {
        error: error.message,
        message: 'Safety proof integration failed'
      });
    }
  }

  async testRealTimeSafetyStatus() {
    try {
      // Test temporal proofs with real data
      const summaryResponse = await axios.get(`${BACKEND_URL}/api/proofs/summary?window=24h`);
      const data = summaryResponse.data;

      const checks = [
        { name: 'NBBO Freshness', value: data.nbboFreshPct, threshold: 95, unit: '%' },
        { name: 'Friction ≤20%', value: data.friction20Pct, threshold: 90, unit: '%' },
        { name: 'Friction ≤25%', value: data.friction25Pct, threshold: 100, unit: '%' },
        { name: 'Cap Violations', value: data.capViolations, threshold: 0, unit: 'count', invert: true }
      ];

      let allPassed = true;
      const statusDetails = {};

      for (const check of checks) {
        const passed = check.invert ? check.value <= check.threshold : check.value >= check.threshold;
        statusDetails[check.name] = {
          value: check.value,
          threshold: check.threshold,
          passed: passed
        };
        if (!passed) allPassed = false;
      }

      this.addTest('Real-Time Safety Status', allPassed, {
        ...statusDetails,
        message: allPassed ? 'All safety metrics within thresholds' : 'Some safety metrics outside acceptable ranges'
      });

    } catch (error) {
      this.addTest('Real-Time Safety Status', false, {
        error: error.message,
        message: 'Cannot determine real-time safety status'
      });
    }
  }

  async runAllTests() {
    this.log('🚀 STARTING END-TO-END SAFETY PROOF TESTING');
    this.log('='.repeat(60));

    // Test service availability
    this.log('🔍 Testing Service Availability...');
    await this.testBackendHealth();
    await this.testFrontendHealth();

    // Test backend APIs
    this.log('🔧 Testing Backend APIs...');
    await this.testSafetyProofEndpoints();
    await this.testOptionsTradingAPI();

    // Test integration
    this.log('🔗 Testing Backend-Frontend Integration...');
    await this.testSafetyProofIntegration();
    await this.testRealTimeSafetyStatus();

    // Generate report
    this.generateReport();
  }

  generateReport() {
    const duration = (new Date() - this.startTime) / 1000;
    const passRate = ((this.results.summary.passed / this.results.summary.total) * 100).toFixed(1);

    this.log('');
    this.log('='.repeat(60));
    this.log('📊 END-TO-END TESTING RESULTS');
    this.log('='.repeat(60));
    this.log(`Total Tests: ${this.results.summary.total}`);
    this.log(`Passed: ${this.results.summary.passed} ✅`);
    this.log(`Failed: ${this.results.summary.failed} ❌`);
    this.log(`Pass Rate: ${passRate}%`);
    this.log(`Duration: ${duration.toFixed(1)}s`);
    this.log('');

    if (this.results.summary.failed === 0) {
      this.log('🎉 ALL TESTS PASSED - SYSTEM IS HEALTHY!', 'success');
    } else {
      this.log('⚠️  SOME TESTS FAILED - CHECK SYSTEM STATUS', 'warning');

      this.log('');
      this.log('Failed Tests:');
      this.results.tests.filter(t => !t.result).forEach(test => {
        this.log(`• ${test.name}`, 'error');
        if (test.details.message) {
          console.log(`  ${test.details.message}`);
        }
      });
    }

    this.log('');
    this.log('🔗 Access Points:');
    this.log(`• Frontend UI: ${FRONTEND_URL}`);
    this.log(`• Backend API: ${BACKEND_URL}`);
    this.log(`• Safety Proofs: ${BACKEND_URL}/api/proofs/summary?window=24h`);
    this.log(`• Health Check: ${BACKEND_URL}/api/health`);
  }
}

// Run the tests
if (require.main === module) {
  const tester = new EndToEndTester();
  tester.runAllTests().catch(error => {
    console.error('Testing failed:', error);
    process.exit(1);
  });
}

module.exports = EndToEndTester;
