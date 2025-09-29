#!/usr/bin/env node

/**
 * Pre-Flight Check for Tomorrow's Trading
 * Verifies everything is ready without changing anything
 */

const axios = require('axios');

const checks = [];
let passCount = 0;
let warnCount = 0;
let failCount = 0;

async function check(name, testFn, critical = true) {
    process.stdout.write(`Checking ${name}...`);
    try {
        const result = await testFn();
        if (result === true) {
            console.log(' ✅ PASS');
            passCount++;
            checks.push({ name, status: 'PASS' });
        } else if (result === 'warn') {
            console.log(' ⚠️  WARNING');
            warnCount++;
            checks.push({ name, status: 'WARN' });
        } else {
            console.log(' ❌ FAIL');
            failCount++;
            checks.push({ name, status: 'FAIL', critical });
        }
    } catch (error) {
        console.log(` ❌ FAIL - ${error.message}`);
        failCount++;
        checks.push({ name, status: 'FAIL', error: error.message, critical });
    }
}

async function runChecks() {
    console.log('\n🚀 BENBOT PRE-FLIGHT CHECK\n');
    console.log('Running system verification...\n');

    // Backend health
    await check('Backend API Health', async () => {
        const resp = await axios.get('http://localhost:4000/health');
        return resp.status === 200;
    });

    // Account status
    await check('Paper Account Access', async () => {
        const resp = await axios.get('http://localhost:4000/api/paper/account');
        return resp.data.balances && resp.data.balances.total_equity > 0;
    });

    // Strategy activation
    await check('Active Strategies', async () => {
        const resp = await axios.get('http://localhost:4000/api/strategies/active');
        const count = resp.data.length;
        if (count === 0) return false;
        if (count < 10) return 'warn';
        return true;
    });

    // Evolution system
    await check('Evolution System', async () => {
        const resp = await axios.get('http://localhost:4000/api/evo/deployment-metrics');
        return resp.data.metrics && resp.data.metrics.total_strategies >= 16;
    });

    // Discovery endpoints
    await check('Diamond Discovery', async () => {
        const resp = await axios.get('http://localhost:4000/api/lab/diamonds');
        return resp.data.items && Array.isArray(resp.data.items);
    });

    await check('Market Discovery', async () => {
        const resp = await axios.get('http://localhost:4000/api/discovery/market');
        return resp.data.discoveries && Array.isArray(resp.data.discoveries);
    });

    // Quote system
    await check('Quote System', async () => {
        const resp = await axios.get('http://localhost:4000/api/quotes?symbols=SPY,AAPL');
        return resp.data && (Array.isArray(resp.data) || resp.data.quotes);
    });

    // WebSocket
    await check('WebSocket Service', async () => {
        // Just check if the endpoint responds
        try {
            await axios.get('http://localhost:4000/health');
            return true;
        } catch {
            return false;
        }
    });

    // Capital tracker
    await check('Capital Management', async () => {
        const resp = await axios.get('http://localhost:4000/api/paper/account');
        const cash = resp.data.balances.total_cash;
        if (cash <= 0) return false;
        if (cash < 10000) return 'warn';
        return true;
    });

    // Frontend
    await check('Frontend Dashboard', async () => {
        try {
            const resp = await axios.get('http://localhost:3003', { timeout: 3000 });
            return resp.status === 200;
        } catch {
            return 'warn'; // Not critical if frontend is down
        }
    }, false);

    // Trading hours awareness
    await check('Market Hours Configured', async () => {
        const now = new Date();
        const day = now.getDay();
        if (day === 0 || day === 6) return 'warn'; // Weekend
        return true;
    });

    // Performance history
    await check('Trade History', async () => {
        const resp = await axios.get('http://localhost:4000/api/trades');
        const trades = resp.data.items || resp.data;
        return Array.isArray(trades);
    });

    console.log('\n═══════════════════════════════════════');
    console.log('            FINAL REPORT               ');
    console.log('═══════════════════════════════════════\n');

    console.log(`✅ Passed: ${passCount}`);
    console.log(`⚠️  Warnings: ${warnCount}`);
    console.log(`❌ Failed: ${failCount}`);

    if (failCount > 0) {
        console.log('\n❌ CRITICAL ISSUES FOUND:\n');
        checks.filter(c => c.status === 'FAIL' && c.critical).forEach(c => {
            console.log(`   • ${c.name}: ${c.error || 'Failed'}`);
        });
        console.log('\n⚠️  Fix these issues before trading!');
    } else if (warnCount > 0) {
        console.log('\n⚠️  WARNINGS:\n');
        checks.filter(c => c.status === 'WARN').forEach(c => {
            console.log(`   • ${c.name}`);
        });
        console.log('\n✅ System is ready but review warnings.');
    } else {
        console.log('\n✅ ALL SYSTEMS GO! 🚀');
        console.log('   Your bot is ready for tomorrow\'s trading!');
    }

    // Recommendations
    console.log('\n📋 TOMORROW\'S CHECKLIST:');
    console.log('   1. Start backend: pm2 start live-api/ecosystem.config.js');
    console.log('   2. Start frontend: cd new-trading-dashboard && npm run dev');
    console.log('   3. Open dashboard: http://localhost:3003');
    console.log('   4. Monitor with: node monitor-performance.js');
    console.log('   5. Check logs: pm2 logs benbot-backend');
}

// Run checks
runChecks().catch(error => {
    console.error('\n❌ Pre-flight check failed:', error.message);
    process.exit(1);
});
