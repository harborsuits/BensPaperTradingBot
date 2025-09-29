#!/usr/bin/env node

/**
 * Real-Time Performance Monitor
 * Shows live trading performance without modifying anything
 * Run this during trading to watch your bot in action!
 */

const axios = require('axios');
const chalk = require('chalk');

// ANSI escape codes for clearing screen
const clear = () => process.stdout.write('\033[2J\033[0;0H');

async function getPerformanceData() {
    try {
        // Fetch all necessary data
        const [strategies, trades, account, positions, decisions] = await Promise.all([
            axios.get('http://localhost:4000/api/strategies/active'),
            axios.get('http://localhost:4000/api/trades'),
            axios.get('http://localhost:4000/api/paper/account'),
            axios.get('http://localhost:4000/api/paper/positions'),
            axios.get('http://localhost:4000/api/decisions/recent?limit=5')
        ]);

        return {
            strategies: strategies.data,
            trades: trades.data.items || trades.data,
            account: account.data.balances,
            positions: positions.data,
            decisions: decisions.data
        };
    } catch (error) {
        console.error('Error fetching data:', error.message);
        return null;
    }
}

function calculateStats(trades) {
    if (!trades || trades.length === 0) {
        return { winRate: 0, totalPnL: 0, wins: 0, losses: 0, avgWin: 0, avgLoss: 0 };
    }

    const closedTrades = trades.filter(t => t.exit_price);
    const wins = closedTrades.filter(t => t.profit > 0);
    const losses = closedTrades.filter(t => t.profit < 0);

    const totalPnL = closedTrades.reduce((sum, t) => sum + (t.profit || 0), 0);
    const avgWin = wins.length > 0 ? wins.reduce((sum, t) => sum + t.profit, 0) / wins.length : 0;
    const avgLoss = losses.length > 0 ? losses.reduce((sum, t) => sum + t.profit, 0) / losses.length : 0;

    return {
        winRate: closedTrades.length > 0 ? (wins.length / closedTrades.length * 100).toFixed(1) : 0,
        totalPnL: totalPnL.toFixed(2),
        wins: wins.length,
        losses: losses.length,
        avgWin: avgWin.toFixed(2),
        avgLoss: avgLoss.toFixed(2)
    };
}

async function displayDashboard() {
    const data = await getPerformanceData();
    if (!data) return;

    const stats = calculateStats(data.trades);
    const pnlPercent = ((data.account.total_equity - 100000) / 100000 * 100).toFixed(2);
    
    clear();
    
    console.log(chalk.cyan.bold('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan.bold('                    🤖 BENBOT LIVE MONITOR 🤖                   '));
    console.log(chalk.cyan.bold('═══════════════════════════════════════════════════════════════\n'));

    // Account Summary
    console.log(chalk.yellow.bold('📊 ACCOUNT SUMMARY'));
    console.log(chalk.white('─────────────────────────────────────────────────────────────'));
    console.log(`Total Equity: ${chalk.green.bold('$' + data.account.total_equity.toFixed(2))} (${pnlPercent >= 0 ? chalk.green('+') : chalk.red('')}${pnlPercent}%)`);
    console.log(`Cash Available: $${data.account.total_cash.toFixed(2)}`);
    console.log(`Open Positions: ${data.positions.length}`);
    console.log();

    // Performance Stats
    console.log(chalk.yellow.bold('📈 PERFORMANCE STATS'));
    console.log(chalk.white('─────────────────────────────────────────────────────────────'));
    console.log(`Win Rate: ${stats.winRate >= 50 ? chalk.green(stats.winRate + '%') : chalk.red(stats.winRate + '%')}`);
    console.log(`Wins/Losses: ${chalk.green(stats.wins + ' wins')} / ${chalk.red(stats.losses + ' losses')}`);
    console.log(`Total P&L: ${stats.totalPnL >= 0 ? chalk.green('$' + stats.totalPnL) : chalk.red('$' + stats.totalPnL)}`);
    console.log(`Avg Win: ${chalk.green('$' + stats.avgWin)} | Avg Loss: ${chalk.red('$' + stats.avgLoss)}`);
    console.log();

    // Active Strategies
    console.log(chalk.yellow.bold('⚡ ACTIVE STRATEGIES'));
    console.log(chalk.white('─────────────────────────────────────────────────────────────'));
    data.strategies.forEach(strategy => {
        const trades = strategy.trades || 0;
        const status = trades > 0 ? chalk.green('✓') : chalk.gray('○');
        console.log(`${status} ${strategy.name.padEnd(30)} | Trades: ${trades}`);
    });
    console.log();

    // Current Positions
    if (data.positions.length > 0) {
        console.log(chalk.yellow.bold('💼 CURRENT POSITIONS'));
        console.log(chalk.white('─────────────────────────────────────────────────────────────'));
        data.positions.slice(0, 5).forEach(pos => {
            const pnl = (pos.current_price - pos.avg_price) * pos.qty;
            const pnlPercent = ((pos.current_price - pos.avg_price) / pos.avg_price * 100).toFixed(1);
            const pnlColor = pnl >= 0 ? chalk.green : chalk.red;
            console.log(`${pos.symbol.padEnd(6)} | Qty: ${pos.qty.toString().padEnd(4)} | P&L: ${pnlColor(pnlPercent + '%')}`);
        });
        if (data.positions.length > 5) {
            console.log(`... and ${data.positions.length - 5} more positions`);
        }
        console.log();
    }

    // Recent Decisions
    console.log(chalk.yellow.bold('🧠 RECENT DECISIONS'));
    console.log(chalk.white('─────────────────────────────────────────────────────────────'));
    data.decisions.forEach(decision => {
        const time = new Date(decision.timestamp).toLocaleTimeString();
        const actionColor = decision.action === 'BUY' ? chalk.green : 
                          decision.action === 'SELL' ? chalk.red : chalk.gray;
        console.log(`${time} | ${decision.symbol.padEnd(6)} | ${actionColor(decision.action.padEnd(4))} | Score: ${decision.score.toFixed(2)}`);
    });

    // Evolution Progress
    const tradesUntilEvolution = 50 - (data.trades.length % 50);
    console.log();
    console.log(chalk.yellow.bold('🧬 EVOLUTION PROGRESS'));
    console.log(chalk.white('─────────────────────────────────────────────────────────────'));
    console.log(`Next evolution in: ${chalk.cyan(tradesUntilEvolution + ' trades')}`);
    console.log(`Total strategies: ${chalk.cyan(data.strategies.length)} (6 base + ${data.strategies.length - 6} evolved)`);

    console.log('\n' + chalk.gray('Refreshing every 5 seconds... Press Ctrl+C to exit'));
}

// Main loop
async function monitor() {
    console.log(chalk.yellow('Starting BenBot Performance Monitor...'));
    
    // Initial display
    await displayDashboard();
    
    // Refresh every 5 seconds
    setInterval(async () => {
        await displayDashboard();
    }, 5000);
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log(chalk.yellow('\n\nMonitor stopped. Good luck with your trading! 🚀'));
    process.exit(0);
});

// Check if chalk is installed
try {
    require.resolve('chalk');
    monitor().catch(console.error);
} catch(e) {
    console.log('Installing chalk for colored output...');
    require('child_process').execSync('npm install chalk', { stdio: 'inherit' });
    console.log('Please run this script again!');
}
