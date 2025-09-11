#!/usr/bin/env node

/**
 * Build Reaction Statistics Script
 * Computes empirical reaction profiles from historical data
 */

const path = require('path');
const fs = require('fs');
const { ReactionStatsBuilder } = require('../src/services/reactionStatsBuilder');

async function buildReactionStats() {
  console.log('🔬 Building Reaction Statistics from Historical Data');
  console.log('==================================================');

  const builder = new ReactionStatsBuilder();

  try {
    // Build statistics from last 365 days
    await builder.buildReactionStats(365, 50); // Lower threshold for testing

    // Export to JSON file
    const stats = builder.exportStats();
    const outputPath = path.join(__dirname, '../data/reaction-stats.json');

    fs.writeFileSync(outputPath, JSON.stringify(stats, null, 2));
    console.log(`✅ Saved reaction statistics to: ${outputPath}`);

    // Show summary
    const validatedEvents = builder.getValidatedEventTypes();
    console.log('');
    console.log('📊 Summary:');
    console.log(`   Total event types processed: ${Object.keys(stats).length}`);
    console.log(`   Validated event types: ${validatedEvents.length}`);

    if (validatedEvents.length > 0) {
      console.log('');
      console.log('🎯 Validated Events:');
      validatedEvents.forEach(event => {
        console.log(`   ${event.eventType} (${event.sector}): ${event.hitRate.toFixed(1)}% hit rate, ${(event.effectSize * 100).toFixed(1)}% effect`);
      });
    }

    console.log('');
    console.log('✅ Reaction statistics ready for production use!');

    return stats;

  } catch (error) {
    console.error('❌ Failed to build reaction statistics:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  buildReactionStats().then(stats => {
    console.log(`🎉 Build complete! ${Object.keys(stats).length} event types processed.`);
  }).catch(error => {
    console.error('💥 Build failed:', error);
    process.exit(1);
  });
}

module.exports = { buildReactionStats };
