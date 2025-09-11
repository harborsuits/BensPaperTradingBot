#!/usr/bin/env node

/**
 * Create Sample Reaction Statistics
 * Generates validated event types for testing the news system
 */

const fs = require('fs');
const path = require('path');

// Sample validated event types based on real market behavior
const sampleReactionStats = {
  "earnings.surprise.up": {
    "technology": {
      "eventType": "earnings.surprise.up",
      "sector": "technology",
      "sampleSize": 324,
      "sampleSize5m": 312,
      "sampleSize30m": 298,
      "sampleSize1d": 276,
      "avgReturn5m": 0.0032,
      "stdReturn5m": 0.0087,
      "medianReturn5m": 0.0028,
      "skewReturn5m": 0.15,
      "avgReturn30m": 0.0056,
      "stdReturn30m": 0.0123,
      "avgReturn1d": 0.0089,
      "stdReturn1d": 0.0187,
      "hitRate5m": 0.62,
      "hitRate30m": 0.58,
      "hitRate1d": 0.61,
      "effectSize5m": 0.37,
      "effectSize30m": 0.46,
      "effectSize1d": 0.48,
      "lastUpdated": "2025-09-10T04:00:00.000Z",
      "passesValidation": true,
      "orthogonalityScore": 0.18,
      "minSampleThreshold": true,
      "last12mThreshold": true,
      "effectThreshold": true
    },
    "healthcare": {
      "eventType": "earnings.surprise.up",
      "sector": "healthcare",
      "sampleSize": 198,
      "sampleSize5m": 192,
      "sampleSize30m": 185,
      "sampleSize1d": 167,
      "avgReturn5m": 0.0028,
      "stdReturn5m": 0.0076,
      "medianReturn5m": 0.0024,
      "skewReturn5m": 0.22,
      "avgReturn30m": 0.0049,
      "stdReturn30m": 0.0111,
      "avgReturn1d": 0.0072,
      "stdReturn1d": 0.0165,
      "hitRate5m": 0.59,
      "hitRate30m": 0.55,
      "hitRate1d": 0.57,
      "effectSize5m": 0.37,
      "effectSize30m": 0.44,
      "effectSize1d": 0.44,
      "lastUpdated": "2025-09-10T04:00:00.000Z",
      "passesValidation": true,
      "orthogonalityScore": 0.15,
      "minSampleThreshold": true,
      "last12mThreshold": true,
      "effectThreshold": true
    }
  },
  "earnings.surprise.down": {
    "technology": {
      "eventType": "earnings.surprise.down",
      "sector": "technology",
      "sampleSize": 287,
      "sampleSize5m": 278,
      "sampleSize30m": 265,
      "sampleSize1d": 243,
      "avgReturn5m": -0.0041,
      "stdReturn5m": 0.0098,
      "medianReturn5m": -0.0036,
      "skewReturn5m": -0.18,
      "avgReturn30m": -0.0067,
      "stdReturn30m": 0.0134,
      "avgReturn1d": -0.0098,
      "stdReturn1d": 0.0192,
      "hitRate5m": 0.38,
      "hitRate30m": 0.42,
      "hitRate1d": 0.41,
      "effectSize5m": -0.42,
      "effectSize30m": -0.50,
      "effectSize1d": -0.51,
      "lastUpdated": "2025-09-10T04:00:00.000Z",
      "passesValidation": true,
      "orthogonalityScore": 0.22,
      "minSampleThreshold": true,
      "last12mThreshold": true,
      "effectThreshold": true
    }
  },
  "macro.fed.hawkish": {
    "financials": {
      "eventType": "macro.fed.hawkish",
      "sector": "financials",
      "sampleSize": 156,
      "sampleSize5m": 152,
      "sampleSize30m": 148,
      "sampleSize1d": 139,
      "avgReturn5m": -0.0021,
      "stdReturn5m": 0.0067,
      "medianReturn5m": -0.0019,
      "skewReturn5m": -0.12,
      "avgReturn30m": -0.0034,
      "stdReturn30m": 0.0089,
      "avgReturn1d": -0.0048,
      "stdReturn1d": 0.0123,
      "hitRate5m": 0.44,
      "hitRate30m": 0.46,
      "hitRate1d": 0.45,
      "effectSize5m": -0.31,
      "effectSize30m": -0.38,
      "effectSize1d": -0.39,
      "lastUpdated": "2025-09-10T04:00:00.000Z",
      "passesValidation": true,
      "orthogonalityScore": 0.28,
      "minSampleThreshold": true,
      "last12mThreshold": true,
      "effectThreshold": true
    }
  },
  "macro.fed.dovish": {
    "technology": {
      "eventType": "macro.fed.dovish",
      "sector": "technology",
      "sampleSize": 134,
      "sampleSize5m": 131,
      "sampleSize30m": 127,
      "sampleSize1d": 118,
      "avgReturn5m": 0.0018,
      "stdReturn5m": 0.0059,
      "medianReturn5m": 0.0016,
      "skewReturn5m": 0.08,
      "avgReturn30m": 0.0029,
      "stdReturn30m": 0.0078,
      "avgReturn1d": 0.0041,
      "stdReturn1d": 0.0109,
      "hitRate5m": 0.56,
      "hitRate30m": 0.53,
      "hitRate1d": 0.54,
      "effectSize5m": 0.31,
      "effectSize30m": 0.37,
      "effectSize1d": 0.38,
      "lastUpdated": "2025-09-10T04:00:00.000Z",
      "passesValidation": true,
      "orthogonalityScore": 0.25,
      "minSampleThreshold": true,
      "last12mThreshold": true,
      "effectThreshold": true
    }
  },
  "corporate.mna.deal": {
    "technology": {
      "eventType": "corporate.mna.deal",
      "sector": "technology",
      "sampleSize": 89,
      "sampleSize5m": 87,
      "sampleSize30m": 84,
      "sampleSize1d": 79,
      "avgReturn5m": 0.0037,
      "stdReturn5m": 0.0089,
      "medianReturn5m": 0.0032,
      "skewReturn5m": 0.19,
      "avgReturn30m": 0.0058,
      "stdReturn30m": 0.0112,
      "avgReturn1d": 0.0079,
      "stdReturn1d": 0.0156,
      "hitRate5m": 0.61,
      "hitRate30m": 0.59,
      "hitRate1d": 0.60,
      "effectSize5m": 0.42,
      "effectSize30m": 0.52,
      "effectSize1d": 0.51,
      "lastUpdated": "2025-09-10T04:00:00.000Z",
      "passesValidation": true,
      "orthogonalityScore": 0.21,
      "minSampleThreshold": true,
      "last12mThreshold": true,
      "effectThreshold": true
    }
  }
};

// Create the data directory if it doesn't exist
const dataDir = path.join(__dirname, '../data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

// Write the sample data
const outputPath = path.join(dataDir, 'reaction-stats.json');
fs.writeFileSync(outputPath, JSON.stringify(sampleReactionStats, null, 2));

console.log('✅ Created sample reaction statistics!');
console.log(`📁 Saved to: ${outputPath}`);
console.log('');
console.log('📊 Sample includes:');
console.log('   • 5 validated event types');
console.log('   • Multiple sectors (technology, healthcare, financials)');
console.log('   • Realistic effect sizes and hit rates');
console.log('   • All statistical validation gates passed');
console.log('');
console.log('🎯 Ready for testing the news system!');
console.log('   Run: curl localhost:4000/api/news/status');

module.exports = { sampleReactionStats };
