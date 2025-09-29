#!/usr/bin/env node

/**
 * Safe Multi-Timeframe Test Script
 * Run this AFTER tomorrow's successful test to see multi-timeframe in action
 * WITHOUT modifying any production code
 */

const axios = require('axios');

async function testMultiTimeframe(symbol = 'SPY') {
    console.log(`\n=== Testing Multi-Timeframe Analysis for ${symbol} ===\n`);
    
    try {
        // Get data for different timeframes
        const timeframes = ['5min', '15min', '1hour', 'day'];
        const data = {};
        
        for (const tf of timeframes) {
            console.log(`Fetching ${tf} data...`);
            const response = await axios.get(`http://localhost:4000/api/bars`, {
                params: {
                    symbol: symbol,
                    timeframe: tf === '5min' ? '5Min' : 
                              tf === '15min' ? '15Min' :
                              tf === '1hour' ? '1Hour' : '1Day',
                    limit: 20
                }
            });
            
            if (response.data && response.data.length > 0) {
                const latest = response.data[response.data.length - 1];
                const prev = response.data[response.data.length - 2];
                
                data[tf] = {
                    close: latest.c,
                    trend: latest.c > prev.c ? 'UP' : 'DOWN',
                    change: ((latest.c - prev.c) / prev.c * 100).toFixed(2) + '%',
                    volume: latest.v
                };
            }
        }
        
        // Multi-timeframe analysis
        console.log('\n📊 Multi-Timeframe Analysis:\n');
        console.log(`Symbol: ${symbol}`);
        console.log('─'.repeat(50));
        
        for (const [tf, info] of Object.entries(data)) {
            console.log(`${tf.padEnd(10)} | Price: $${info.close} | Trend: ${info.trend} | Change: ${info.change}`);
        }
        
        // Trading decision based on multiple timeframes
        console.log('\n🎯 Multi-Timeframe Signal:\n');
        
        const signals = {
            '5min': data['5min']?.trend,
            '15min': data['15min']?.trend,
            '1hour': data['1hour']?.trend,
            'day': data['day']?.trend
        };
        
        const upCount = Object.values(signals).filter(s => s === 'UP').length;
        const totalCount = Object.values(signals).filter(s => s).length;
        
        if (upCount === totalCount) {
            console.log('✅ STRONG BUY - All timeframes aligned UP');
        } else if (upCount >= totalCount * 0.75) {
            console.log('✅ BUY - Most timeframes aligned UP');
        } else if (upCount <= totalCount * 0.25) {
            console.log('🔴 SELL - Most timeframes aligned DOWN');
        } else {
            console.log('⚠️  HOLD - Mixed signals across timeframes');
        }
        
        console.log('\n💡 Benefits of Multi-Timeframe:');
        console.log('   • Confirms trends across different time horizons');
        console.log('   • Reduces false signals from noise');
        console.log('   • Better entry/exit timing');
        console.log('   • Aligns with institutional trading practices');
        
        console.log('\n🔧 To implement in strategies:');
        console.log('   1. Add timeframe array to each strategy');
        console.log('   2. Fetch data for all timeframes');
        console.log('   3. Only trade when timeframes align');
        console.log('   4. Use longer timeframes for trend, shorter for timing');
        
    } catch (error) {
        console.error('Error:', error.message);
        console.log('\n⚠️  Make sure the backend is running!');
    }
}

// Run test
const symbol = process.argv[2] || 'SPY';
testMultiTimeframe(symbol).catch(console.error);
