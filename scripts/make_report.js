#!/usr/bin/env node

/**
 * Evidence Report Generator - Creates comprehensive system assessment
 */

const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:4000';

function makeRequest(path) {
  return new Promise((resolve, reject) => {
    const http = require('http');
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

async function loadEvidenceFile(filename) {
  try {
    const filepath = path.join('evidence', filename);
    if (fs.existsSync(filepath)) {
      const data = fs.readFileSync(filepath, 'utf8');
      return JSON.parse(data);
    }
  } catch (e) {
    // Ignore errors
  }
  return null;
}

function formatCurrency(amount) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2
  }).format(amount || 0);
}

function formatPercent(value) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

async function generateReport() {
  console.log('📊 Generating Comprehensive Evidence Report...\n');

  const report = {
    generated: new Date().toISOString(),
    title: 'BenBot Trading System - Evidence-Based Assessment',
    sections: []
  };

  // 1. System Health
  console.log('🔍 Gathering system health data...');
  const health = await loadEvidenceFile('health.json');
  if (health) {
    report.sections.push({
      title: '🩺 SYSTEM HEALTH',
      data: {
        status: health.ok ? '✅ OPERATIONAL' : '❌ ISSUES',
        breaker: health.breaker || 'UNKNOWN',
        uptime: health.uptime ? `${Math.floor(health.uptime / 3600)}h ${Math.floor((health.uptime % 3600) / 60)}m` : 'UNKNOWN',
        version: health.version || 'UNKNOWN',
        services: health.services || {}
      }
    });
  }

  // 2. Strategy Performance
  console.log('📈 Analyzing strategy performance...');
  const strategies = await loadEvidenceFile('strategies.json');
  if (strategies && strategies.items) {
    const strategyTable = strategies.items.map(s => ({
      name: s.name || s.id,
      status: s.status || 'unknown',
      winRate: formatPercent(s.performance?.win_rate),
      sharpe: (s.performance?.sharpe_ratio || 0).toFixed(2),
      trades: s.performance?.trades_count || 0,
      drawdown: formatPercent(s.performance?.max_drawdown),
      edge: assessEdge(s.performance)
    }));

    report.sections.push({
      title: '🎯 STRATEGY PERFORMANCE',
      data: {
        summary: `${strategies.items.length} strategies loaded`,
        active: strategies.items.filter(s => s.status === 'active').length,
        table: strategyTable
      }
    });
  }

  // 3. Paper Account Status
  console.log('💰 Checking paper account...');
  const paperAccount = await loadEvidenceFile('paper_account.json');
  if (paperAccount) {
    report.sections.push({
      title: '💼 PAPER ACCOUNT STATUS',
      data: {
        balance: formatCurrency(paperAccount.balances?.total_equity),
        cash: formatCurrency(paperAccount.balances?.total_cash),
        margin: {
          buying_power: formatCurrency(paperAccount.balances?.margin?.stock_buying_power),
          maintenance_call: paperAccount.balances?.margin?.maintenance_call || 0
        }
      }
    });
  }

  // 4. Recent Trades
  console.log('📊 Analyzing recent trades...');
  const trades = await loadEvidenceFile('trades_found.json');
  if (trades && trades.items) {
    const recentTrades = trades.items.slice(0, 10);
    const tradeStats = {
      total: trades.items.length,
      symbols: [...new Set(trades.items.map(t => t.symbol))],
      avgQuantity: trades.items.reduce((sum, t) => sum + (t.qty || 0), 0) / trades.items.length,
      timeRange: recentTrades.length > 0 ? {
        from: recentTrades[recentTrades.length - 1].ts?.substring(0, 10),
        to: recentTrades[0].ts?.substring(0, 10)
      } : null
    };

    report.sections.push({
      title: '📈 RECENT TRADING ACTIVITY',
      data: {
        stats: tradeStats,
        sample: recentTrades.slice(0, 3).map(t => ({
          symbol: t.symbol,
          side: t.side,
          quantity: t.qty,
          status: t.status,
          time: t.ts?.substring(0, 19)
        }))
      }
    });
  }

  // 5. Coordination Results
  console.log('🎯 Checking coordination results...');
  const coordination = await loadEvidenceFile('__api_audit_coordination.json');
  if (coordination && coordination.success) {
    report.sections.push({
      title: '🎯 STRATEGY COORDINATION',
      data: {
        lastCycle: coordination.audit ? {
          signals: coordination.audit.rawSignals || 0,
          winners: coordination.audit.winners || 0,
          conflicts: coordination.audit.conflicts || 0,
          rejections: coordination.audit.rejects || 0
        } : 'No recent coordination cycles'
      }
    });
  }

  // 6. Risk Management
  console.log('🛡️ Analyzing risk management...');
  const riskRejections = await loadEvidenceFile('__api_audit_risk-rejections.json');
  if (riskRejections && riskRejections.success) {
    const rejections = riskRejections.rejections || [];
    const rejectionStats = {};
    rejections.forEach(r => {
      rejectionStats[r.reason] = (rejectionStats[r.reason] || 0) + 1;
    });

    report.sections.push({
      title: '🛡️ RISK MANAGEMENT',
      data: {
        totalRejections: rejections.length,
        rejectionReasons: rejectionStats,
        recentRejections: rejections.slice(0, 3).map(r => ({
          reason: r.reason,
          symbol: r.symbol,
          check: r.check,
          time: r.timestamp?.substring(0, 19)
        }))
      }
    });
  }

  // 7. Capital Allocation
  console.log('💰 Checking capital allocation...');
  const allocations = await loadEvidenceFile('__api_audit_allocations_current.json');
  if (allocations && allocations.success) {
    report.sections.push({
      title: '💰 CAPITAL ALLOCATION',
      data: {
        status: allocations.current_allocation ? 'Active' : 'No current allocations',
        allocations: allocations.current_allocation?.allocations || {}
      }
    });
  }

  // 8. Autopilot Status
  console.log('🤖 Checking autopilot status...');
  const autopilot = await loadEvidenceFile('__api_audit_autoloop_status.json');
  if (autopilot && autopilot.success) {
    report.sections.push({
      title: '🤖 AUTOPILOT STATUS',
      data: {
        isRunning: autopilot.autoloop_status?.is_running || false,
        lastRun: autopilot.autoloop_status?.last_run?.substring(0, 19) || 'Never',
        status: autopilot.autoloop_status?.status || 'UNKNOWN'
      }
    });
  }

  // Generate markdown report
  let markdown = `# ${report.title}\n\n`;
  markdown += `**Generated:** ${new Date(report.generated).toLocaleString()}\n\n`;
  markdown += `## Executive Summary\n\n`;
  markdown += `This report provides evidence-based assessment of the BenBot trading system's current state, performance, and capabilities.\n\n`;

  for (const section of report.sections) {
    markdown += `## ${section.title}\n\n`;

    if (section.data) {
      if (section.title.includes('STRATEGY PERFORMANCE') && section.data.table) {
        markdown += `| Strategy | Status | Win Rate | Sharpe | Trades | Max DD | Edge |\n`;
        markdown += `|----------|--------|----------|--------|--------|--------|------|\n`;
        for (const row of section.data.table) {
          markdown += `| ${row.name} | ${row.status} | ${row.winRate} | ${row.sharpe} | ${row.trades} | ${row.drawdown} | ${row.edge} |\n`;
        }
        markdown += `\n`;
      } else {
        markdown += `\`\`\`json\n${JSON.stringify(section.data, null, 2)}\n\`\`\`\n\n`;
      }
    }
  }

  // Write report
  fs.writeFileSync('EVIDENCE_REPORT.md', markdown);
  console.log('✅ Evidence report generated: EVIDENCE_REPORT.md');

  // Print key findings to console
  console.log('\n🎯 KEY FINDINGS:');
  console.log('='.repeat(50));

  if (health && health.ok) {
    console.log('✅ System Health: OPERATIONAL');
  }

  if (strategies && strategies.items) {
    const activeStrategies = strategies.items.filter(s => s.status === 'active');
    console.log(`✅ Active Strategies: ${activeStrategies.length}/${strategies.items.length}`);

    const topStrategy = strategies.items.find(s => s.performance?.win_rate > 0.5);
    if (topStrategy) {
      console.log(`🎯 Best Performing: ${topStrategy.name} (${formatPercent(topStrategy.performance.win_rate)} win rate)`);
    }
  }

  if (paperAccount) {
    console.log(`💰 Paper Account: ${formatCurrency(paperAccount.balances?.total_equity)}`);
  }

  if (trades && trades.items) {
    console.log(`📊 Recent Trades: ${trades.items.length} transactions`);
  }

  console.log('\n📋 Report saved as: EVIDENCE_REPORT.md');
}

function assessEdge(performance) {
  if (!performance) return '❓';

  const winRate = performance.win_rate || 0;
  const sharpe = performance.sharpe_ratio || 0;
  const trades = performance.trades_count || 0;

  if (winRate > 0.55 && sharpe > 1.0 && trades > 50) {
    return '✅ STRONG';
  } else if (winRate > 0.52 && sharpe > 0.5) {
    return '⚠️ MODERATE';
  } else if (winRate > 0.5) {
    return '🤔 WEAK';
  } else {
    return '❌ NONE';
  }
}

// Ensure evidence directory exists
if (!fs.existsSync('evidence')) {
  fs.mkdirSync('evidence');
}

generateReport().catch(console.error);
