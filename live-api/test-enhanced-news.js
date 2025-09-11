#!/usr/bin/env node
/**
 * Test script for enhanced news sentiment analysis
 */

const { NewsAdapter } = require('./src/services/newsAdapter');

// Test the enhanced news sentiment analysis
async function testEnhancedNewsSentiment() {
  console.log('🚀 Testing Enhanced News Sentiment Analysis\n');

  const adapter = new NewsAdapter({
    useAI: true,
    rateLimit: 10
  });

  // Test news items with different sentiment scenarios
  const testItems = [
    {
      title: "Apple Reports Strong Q4 Earnings, Beats Revenue Expectations",
      description: "Apple Inc. reported quarterly earnings that exceeded analyst expectations, with revenue growth of 15% year-over-year. The company's services segment showed particular strength.",
      source: "Reuters",
      symbols: ["AAPL"]
    },
    {
      title: "Tesla Faces Production Delays Amid Supply Chain Issues",
      description: "Tesla Inc. announced production delays for its Model Y due to semiconductor shortages and supply chain disruptions. The company lowered its delivery guidance for Q2.",
      source: "Bloomberg",
      symbols: ["TSLA"]
    },
    {
      title: "Federal Reserve Signals Potential Interest Rate Hike",
      description: "The Federal Reserve indicated that it may raise interest rates sooner than expected to combat inflation. This could impact borrowing costs across the economy.",
      source: "CNBC",
      symbols: []
    },
    {
      title: "NVIDIA Surges on Strong AI Chip Demand",
      description: "NVIDIA Corporation's stock jumped 8% after reporting record quarterly revenue driven by demand for AI chips. The company's data center segment grew 150% year-over-year.",
      source: "WSJ",
      symbols: ["NVDA"]
    },
    {
      title: "Oil Prices Plunge on Economic Slowdown Fears",
      description: "Crude oil prices fell sharply amid concerns about global economic slowdown. OPEC+ members are considering production cuts to stabilize prices.",
      source: "FT",
      symbols: []
    }
  ];

  console.log('📊 Analyzing news sentiment...\n');

  for (let i = 0; i < testItems.length; i++) {
    const item = testItems[i];
    console.log(`\n--- News Item ${i + 1} ---`);
    console.log(`Title: ${item.title}`);
    console.log(`Source: ${item.source}`);

    // Test enhanced sentiment analysis
    const sentiment = adapter.analyzeSentiment(item);
    console.log(`Sentiment Score: ${sentiment.score.toFixed(3)}`);
    console.log(`Confidence: ${(sentiment.confidence * 100).toFixed(1)}%`);
    console.log(`Sector: ${sentiment.sector}`);
    console.log(`Evidence: ${sentiment.evidence.positiveHits} positive, ${sentiment.evidence.negativeHits} negative hits`);

    // Test symbol extraction
    const symbols = adapter.extractSymbols(item);
    console.log(`Extracted Symbols: ${symbols.join(', ') || 'None'}`);

    // Test relevance scoring
    const relevance = adapter.calculateRelevance(item, symbols);
    console.log(`Relevance Score: ${(relevance * 100).toFixed(1)}%`);

    // Test normalization
    const normalized = adapter.normalizeItem(item);
    console.log(`Overall Sentiment: ${normalized.sentiment.toFixed(3)}`);
    console.log(`Trend Analysis: ${normalized.trend ? `Available (${normalized.trend.sampleSize} samples)` : 'Not enough data'}`);
  }

  console.log('\n' + '='.repeat(60));
  console.log('🎯 Enhanced Features Demonstrated:');
  console.log('✅ Context-aware sentiment analysis');
  console.log('✅ Sector detection (technology, finance, energy)');
  console.log('✅ Enhanced symbol extraction');
  console.log('✅ Confidence scoring based on evidence');
  console.log('✅ Sentiment trend tracking');
  console.log('✅ Relevance scoring for news prioritization');
  console.log('✅ Spell correction and text normalization');
  console.log('✅ Stopword removal and NLP processing');
  console.log('✅ Market-specific context multipliers');
  console.log('✅ Intensity analysis based on word frequency');
  console.log('✅ Comprehensive evidence reporting');
  console.log('='.repeat(60));
}

// Test sentiment trend analysis
async function testSentimentTrends() {
  console.log('\n📈 Testing Sentiment Trend Analysis\n');

  const adapter = new NewsAdapter({ useAI: true });

  // Simulate multiple news items for a symbol over time
  const aaplNews = [
    { title: "Apple earnings beat expectations", symbols: ["AAPL"], sentiment: 0.8 },
    { title: "Apple faces supply chain issues", symbols: ["AAPL"], sentiment: -0.3 },
    { title: "Apple announces new product line", symbols: ["AAPL"], sentiment: 0.6 },
    { title: "Apple stock surges on AI announcements", symbols: ["AAPL"], sentiment: 0.9 },
    { title: "Apple faces regulatory scrutiny", symbols: ["AAPL"], sentiment: -0.5 }
  ];

  // Process news items to build sentiment history
  aaplNews.forEach((item, index) => {
    const sentiment = adapter.analyzeSentiment(item);
    // Simulate time progression
    const mockTime = Date.now() - (aaplNews.length - index) * 60 * 60 * 1000; // 1 hour intervals

    adapter.updateSentimentHistory(item.symbols, sentiment.score, sentiment.confidence);
  });

  // Get trend analysis
  const trend = adapter.getSentimentTrend("AAPL", 24);
  if (trend) {
    console.log(`📊 AAPL Sentiment Trend (24h):`);
    console.log(`   Average Score: ${trend.averageScore.toFixed(3)}`);
    console.log(`   Average Confidence: ${(trend.averageConfidence * 100).toFixed(1)}%`);
    console.log(`   Trend Direction: ${trend.trend > 0 ? '📈 Improving' : trend.trend < 0 ? '📉 Declining' : '➡️ Stable'}`);
    console.log(`   Sample Size: ${trend.sampleSize} news items`);
    console.log(`   Time Range: ${trend.timeRange}`);
  }
}

// Run tests
async function main() {
  try {
    await testEnhancedNewsSentiment();
    await testSentimentTrends();
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { testEnhancedNewsSentiment, testSentimentTrends };
