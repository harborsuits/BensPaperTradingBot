#!/usr/bin/env node

/**
 * News System Integration Script
 * Integrates EventClassifier, ReactionStatsBuilder, and NewsNudge into the main server
 */

const fs = require('fs');
const path = require('path');

// Import the new components
const { EventClassifier } = require('../src/services/eventClassifier');
const { ReactionStatsBuilder } = require('../src/services/reactionStatsBuilder');
const { NewsNudge } = require('../src/services/newsNudge');

// Server file path
const serverPath = path.join(__dirname, '../server.js');

// Read the current server file
let serverContent = fs.readFileSync(serverPath, 'utf8');

// Integration points to add
const integrations = [
  // 1. Add imports at the top
  {
    search: "const { recorder } = require('./src/services/marketRecorder.js');",
    replace: `const { recorder } = require('./src/services/marketRecorder.js');

// News System Integration
const { EventClassifier } = require('./src/services/eventClassifier');
const { ReactionStatsBuilder } = require('./src/services/reactionStatsBuilder');
const { NewsNudge } = require('./src/services/newsNudge');`
  },

  // 2. Initialize components after other services
  {
    search: "const recorder = new MarketRecorder();",
    replace: `const recorder = new MarketRecorder();

// Initialize News System Components
const eventClassifier = new EventClassifier();
const reactionStatsBuilder = new ReactionStatsBuilder();
const newsNudge = new NewsNudge(reactionStatsBuilder);

// Load pre-computed reaction statistics (if available)
try {
  const statsPath = path.join(__dirname, '../data/reaction-stats.json');
  if (fs.existsSync(statsPath)) {
    reactionStatsBuilder.loadStatsFromFile(statsPath);
    console.log('📊 Loaded pre-computed reaction statistics');
  }
} catch (error) {
  console.log('⚠️  Could not load reaction statistics:', error.message);
}`
  },

  // 3. Add news events endpoint
  {
    search: "app.get('/api/context/news', (req, res) => {",
    replace: `// News Events with Event Classification
app.get('/api/news/events', async (req, res) => {
  try {
    const headlines = req.query.headlines ? JSON.parse(req.query.headlines) : [];
    const source = req.query.source || 'unknown';
    const tickers = req.query.tickers ? req.query.tickers.split(',') : [];

    if (headlines.length === 0) {
      return res.json({
        events: [],
        message: 'No headlines provided',
        asOf: asOf()
      });
    }

    // Classify each headline
    const allEvents = [];
    for (const headline of headlines) {
      const events = eventClassifier.classify(headline, source, tickers);
      allEvents.push(...events);
    }

    // Validate events against reaction statistics
    const validatedEvents = allEvents.map(event => {
      const sector = 'technology'; // Simplified - would derive from tickers
      const isValid = newsNudge.validateEvent(event, sector);
      return { ...event, validated: isValid };
    });

    res.json({
      events: validatedEvents,
      totalHeadlines: headlines.length,
      totalEvents: allEvents.length,
      validatedEvents: validatedEvents.filter(e => e.validated).length,
      classifierStats: eventClassifier.getPerformanceStats(),
      asOf: asOf()
    });

  } catch (error) {
    console.error('News events error:', error);
    res.status(500).json({
      error: error.message,
      events: [],
      asOf: asOf()
    });
  }
});

// Legacy news context (enhanced with events)
app.get('/api/context/news', (req, res) => {`
  },

  // 4. Add news nudge endpoint
  {
    search: "app.get('/api/context/sentiment/anomalies', (req, res) => {",
    replace: `app.get('/api/context/sentiment/anomalies', (req, res) => {
  res.json({ anomalies: [], asOf: asOf() });
});

// News Nudge for Trading Plans
app.post('/api/context/news-nudge', async (req, res) => {
  try {
    const { events, marketContext, sector, symbol } = req.body;

    if (!events || !Array.isArray(events)) {
      return res.status(400).json({
        error: 'Events array required',
        nudge: 0,
        asOf: asOf()
      });
    }

    // Validate and calculate nudge
    const nudge = newsNudge.calculateNudge(events, marketContext || {}, sector || 'technology', symbol);

    // Get explanation for UI
    const explanation = newsNudge.getNudgeExplanation(nudge, events, marketContext || {});

    res.json({
      nudge,
      explanation,
      circuitBreaker: newsNudge.getCircuitBreakerStatus(),
      performance: newsNudge.getPerformanceStats(),
      asOf: asOf()
    });

  } catch (error) {
    console.error('News nudge error:', error);
    res.status(500).json({
      error: error.message,
      nudge: 0,
      explanation: { reason: 'Error calculating nudge' },
      asOf: asOf()
    });
  }
});

// News System Status
app.get('/api/news/status', (req, res) => {
  res.json({
    classifier: eventClassifier.getPerformanceStats(),
    nudge: newsNudge.getPerformanceStats(),
    validatedEvents: reactionStatsBuilder.getValidatedEventTypes(),
    circuitBreaker: newsNudge.getCircuitBreakerStatus(),
    asOf: asOf()
  });
});`
  },

  // 5. Enhance context endpoint with news data
  {
    search: "app.get('/api/context', (req, res) => {",
    replace: `app.get('/api/context', (req, res) => {
  // Get recent news events for context
  const recentEvents = reactionStatsBuilder.getValidatedEventTypes();

  res.json({
    timestamp: asOf(),
    regime: { type: 'Neutral', confidence: 0.58, description: 'Mixed breadth, range-bound.' },
    volatility: { value: 17.2, change: -0.3, classification: 'Medium' },
    sentiment: { score: 0.52, sources: ['news', 'social'], trending_words: ['AI', 'earnings', 'CPI'] },
    news: {
      validatedEventTypes: recentEvents.length,
      circuitBreakerActive: newsNudge.getCircuitBreakerStatus().active,
      lastEventClassification: eventClassifier.getPerformanceStats()
    },
    features: { vix: 17.2, put_call: 0.92, adv_dec: 1.1 },
  });
});`
  }
];

// Apply integrations
let modifiedContent = serverContent;
for (const integration of integrations) {
  if (modifiedContent.includes(integration.search)) {
    modifiedContent = modifiedContent.replace(integration.search, integration.replace);
  } else {
    console.log(`⚠️  Could not find integration point: ${integration.search.substring(0, 50)}...`);
  }
}

// Write back the modified server file
fs.writeFileSync(serverPath, modifiedContent);

console.log('✅ News System Integration Complete!');
console.log('');
console.log('📋 New Endpoints Added:');
console.log('  POST /api/context/news-nudge - Calculate confidence nudge from events');
console.log('  GET  /api/news/events - Classify headlines into events');
console.log('  GET  /api/news/status - News system performance stats');
console.log('');
console.log('🔧 Enhanced Endpoints:');
console.log('  GET  /api/context - Now includes news system status');
console.log('');
console.log('🚀 Next Steps:');
console.log('1. Restart the server: npm start');
console.log('2. Build reaction statistics: node scripts/build-reaction-stats.js');
console.log('3. Test the endpoints with: curl localhost:4000/api/news/status');
console.log('');
console.log('📖 Usage Example:');
console.log(`curl -X POST localhost:4000/api/context/news-nudge \\
  -H "Content-Type: application/json" \\
  -d '{"events": [{"type": "earnings.surprise.up", "confidence": 0.9, "validated": true, "effectZ": 0.3}], "marketContext": {"vix": 18}, "sector": "technology"}'`);
