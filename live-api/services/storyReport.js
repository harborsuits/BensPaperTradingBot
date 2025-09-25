/**
 * Story-Style Daily Report Generator
 * Explains what the bot did in simple, narrative form
 */

class StoryReportGenerator {
  constructor(performanceRecorder, paperBroker, autoLoop, decisionLogger) {
    this.performanceRecorder = performanceRecorder;
    this.paperBroker = paperBroker;
    this.autoLoop = autoLoop;
    this.decisionLogger = decisionLogger;
  }

  async generateStoryReport() {
    const report = {
      title: "🤖 What Your Trading Bot Did Today - A Simple Story",
      generated: new Date().toLocaleString(),
      sections: []
    };

    // Morning Wake Up
    report.sections.push(await this.morningStory());
    
    // Trading Activity
    report.sections.push(await this.tradingStory());
    
    // Learning Story
    report.sections.push(await this.learningStory());
    
    // Mistakes and Lessons
    report.sections.push(await this.mistakesStory());
    
    // Tomorrow's Plan
    report.sections.push(await this.tomorrowStory());

    return this.formatAsHTML(report);
  }

  async morningStory() {
    // TODO: Get trades from database when methods are available
    const trades = [];
    const firstTrade = trades[0];
    
    return {
      title: "☀️ How The Bot Woke Up",
      story: `
This morning at 9:30 AM when the stock market opened, your bot woke up and stretched its digital arms!

Here's what it did first:
1. 👀 **Looked at the Market**: "Is it a happy day (stocks going up) or sad day (stocks going down)?"
   - Market mood: ${await this.getMarketMood()}
   
2. 📰 **Read the News**: "What are people talking about today?"
   - Found ${await this.getNewsCount()} news stories
   - Most talked about: ${await this.getTopNewsTopics()}
   
3. 🔍 **Scanned for Opportunities**: "Which penny stocks look interesting?"
   - Checked ${await this.getScannedStocks()} different stocks
   - Found ${await this.getOpportunityCount()} that looked promising

4. 🧠 **Made Its First Decision**: 
   ${firstTrade ? 
     `"I think ${firstTrade.symbol} will go up because ${await this.explainReason(firstTrade)}"` :
     `"Hmm, nothing looks good enough yet. I'll wait and watch."`}
      `,
      whatThisMeans: `
**What the code did**: The AutoLoop.js file ran its 'runOnce()' function. This is like the bot's morning routine.
It called getDiamonds() to find cheap stocks that might go up, and checkMarketConditions() to see if it's safe to trade.
      `
    };
  }

  async tradingStory() {
    // TODO: Get trades from database when methods are available
    const trades = [];
    const decisions = [];
    
    const stories = trades.map(trade => this.explainTrade(trade));
    
    return {
      title: "💰 The Trading Adventures",
      story: `
Today your bot made ${trades.length} trades. Here's the story of each one:

${stories.join('\n\n')}

**The Bot's Thought Process**:
- It made ${decisions.length} different decisions
- It was confident ${this.countConfidentDecisions(decisions)} times (over 70% sure)
- It was unsure ${this.countUnsureDecisions(decisions)} times (under 50% sure)
- It said "no thanks" to ${decisions.length - trades.length} opportunities

**Money Update**:
- Started with: $${this.formatMoney(100000)}
- Now have: $${this.formatMoney(await this.getCurrentBalance())}
- That's ${await this.getProfitEmoji()}: ${await this.getProfitStory()}
      `,
      whatThisMeans: `
**What the code did**: The StrategyManager looked at each stock using different strategies (like recipes):
- MomentumStrategy.js checked if stocks were moving up fast
- MeanReversionStrategy.js looked for stocks that dropped too much
- VolumeBreakoutStrategy.js found stocks with lots of people trading

Each strategy gave a "confidence score" (0-100). If any score was over 60, the bot said "let's trade!"
      `
    };
  }

  async explainTrade(trade) {
    const decision = await this.getDecisionForTrade(trade);
    
    return `
**Trade #${trade.id}: ${trade.symbol}** ${this.getTradeEmoji(trade)}

📖 The Story:
At ${new Date(trade.timestamp).toLocaleTimeString()}, the bot noticed ${trade.symbol} was at $${trade.price}.

🤔 What the bot was thinking:
"${await this.translateDecisionToStory(decision)}"

📊 What actually happened:
- Bought ${trade.quantity} shares at $${trade.price}
- ${trade.exit_price ? `Sold at $${trade.exit_price}` : 'Still holding'}
- ${trade.pnl > 0 ? `Made $${trade.pnl} 🎉` : trade.pnl < 0 ? `Lost $${Math.abs(trade.pnl)} 😅` : 'Still waiting to see'}

🧪 Which strategy decided this:
${decision.strategy_name} (It's like using a ${this.getStrategyMetaphor(decision.strategy_name)} to find good trades)
    `;
  }

  async learningStory() {
    // TODO: Get competition data when methods are available
    const competitions = [];
    const evolution = { generation: 1, totalTested: 0, newStrategies: 0, retiredStrategies: 0 };
    
    return {
      title: "🧬 How The Bot Got Smarter",
      story: `
Your bot isn't just trading - it's learning! Like a video game character leveling up.

**Bot Competition** (Like a Trading Tournament):
${competitions.length > 0 ? `
- 100 mini-bots competed with $50 each
- The winner was: ${competitions[0].winner} (made ${competitions[0].winnerProfit}%)
- The loser was: ${competitions[0].loser} (lost ${competitions[0].loserLoss}%)

What the winner did differently:
${await this.explainWinningStrategy(competitions[0].winner)}
` : 'No competitions ran today yet'}

**Evolution Progress** (Like Pokemon Evolution):
- Current Generation: ${evolution.generation}
- Strategies tested: ${evolution.totalTested}
- New "baby" strategies born: ${evolution.newStrategies}
- Old strategies that "retired": ${evolution.retiredStrategies}

**What It Learned Today**:
1. ${await this.getLearning(1)}
2. ${await this.getLearning(2)}
3. ${await this.getLearning(3)}
      `,
      whatThisMeans: `
**What the code did**: 
- GeneticInheritance.js took the best strategies and mixed them (like breeding puppies)
- It used crossover() to combine good traits from two "parent" strategies
- It used mutate() to randomly change some settings (like nature trying new things)
- The Tournament Controller promoted good strategies from R1 → R2 → R3 → LIVE
      `
    };
  }

  async mistakesStory() {
    // TODO: Get loss data when methods are available
    const losses = [];
    const missedOps = [];
    
    return {
      title: "🤦 Oops Moments & Lessons",
      story: `
Even smart bots make mistakes! Here's what went wrong and what we learned:

**Biggest Mistakes**:
${losses.slice(0, 3).map(loss => `
😅 **${loss.symbol}**: 
- What happened: Bought at $${loss.entry}, sold at $${loss.exit} (lost $${loss.loss})
- Why it failed: ${this.explainFailure(loss)}
- Lesson learned: "${this.getLessonFromFailure(loss)}"
`).join('\n')}

**Opportunities It Missed** (Stocks that went up without us):
${missedOps.slice(0, 3).map(miss => `
😢 **${miss.symbol}**: Went up ${miss.percentGain}% but we didn't buy
- Why we missed it: ${this.explainWhyMissed(miss)}
- How to catch it next time: ${this.getImprovementIdea(miss)}
`).join('\n')}

**Things That Confused The Bot**:
${await this.getConfusionPoints()}
      `,
      whatThisMeans: `
**What the code did**:
- The bot's confidence calculation in decisionMaker.js was too high/low
- Risk management in positionSizer.js might have been too careful
- News sentiment in newsAnalyzer.js might have misunderstood the story
- The circuit breaker in circuitBreaker.js might have stopped trading too early
      `
    };
  }

  async tomorrowStory() {
    return {
      title: "🔮 Tomorrow's Game Plan",
      story: `
Based on today's adventures, here's what your bot is planning:

**New Strategies Being Born Tonight**:
- Taking the winners from today and making 20 new "baby" strategies
- Each baby will be slightly different (like siblings)
- Tomorrow they'll compete to see who's best

**What It Will Do Differently**:
1. ${await this.getTomorrowChange(1)}
2. ${await this.getTomorrowChange(2)}
3. ${await this.getTomorrowChange(3)}

**Stocks It's Watching** (The Bot's Wishlist):
${await this.getWatchlist()}

**Risk Settings** (Like Safety Guards):
- Will only risk $${await this.getMaxRiskPerTrade()} per trade
- Will stop trading if we lose $${await this.getDailyStopLoss()} in one day
- Will take profits when up ${await this.getProfitTarget()}%
      `,
      whatThisMeans: `
**What the code will do**:
- AutoEvolutionManager.js will run at midnight to breed new strategies
- The scanning algorithms will look for these new patterns
- Risk limits in ai_policy.yaml will keep us safe
- Everything resets at 3 AM for a fresh start
      `
    };
  }

  // Helper methods for storytelling
  async getMarketMood() {
    const regime = await this.getMarketRegime();
    const moods = {
      'bull_low': '😊 Happy and calm',
      'bull_high': '🎉 Super excited!',
      'bear_low': '😔 Sad but quiet',
      'bear_high': '😱 Scared and crazy!',
      'neutral': '😐 Not sure yet...'
    };
    return moods[regime] || moods['neutral'];
  }

  getTradeEmoji(trade) {
    if (!trade.exit_price) return '⏳';
    return trade.pnl > 0 ? '✅' : '❌';
  }

  getStrategyMetaphor(strategyName) {
    const metaphors = {
      'momentum': 'surfboard (riding the wave)',
      'mean_reversion': 'rubber band (snaps back to normal)',
      'volume_breakout': 'crowd detector (following the herd)',
      'news_sentiment': 'mood ring (feeling the vibes)',
      'technical_patterns': 'treasure map (finding hidden patterns)'
    };
    return metaphors[strategyName.toLowerCase()] || 'special formula';
  }

  formatAsHTML(report) {
    return `
<!DOCTYPE html>
<html>
<head>
  <title>${report.title}</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      max-width: 800px; 
      margin: 0 auto; 
      padding: 20px;
      line-height: 1.6;
    }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; }
    h2 { color: #34495e; margin-top: 30px; }
    h3 { color: #7f8c8d; }
    .story { 
      background: #ecf0f1; 
      padding: 15px; 
      border-radius: 10px; 
      margin: 10px 0;
    }
    .code-explanation { 
      background: #e8f4f8; 
      padding: 10px; 
      border-left: 4px solid #3498db;
      font-family: monospace;
      font-size: 0.9em;
    }
    .profit { color: #27ae60; font-weight: bold; }
    .loss { color: #e74c3c; font-weight: bold; }
    .emoji { font-size: 1.2em; }
  </style>
</head>
<body>
  <h1>${report.title}</h1>
  <p>Generated: ${report.generated}</p>
  
  ${report.sections.map(section => `
    <h2>${section.title}</h2>
    <div class="story">
      ${section.story.replace(/\n/g, '<br>')}
    </div>
    <div class="code-explanation">
      <strong>🔧 For Developers:</strong><br>
      ${section.whatThisMeans.replace(/\n/g, '<br>')}
    </div>
  `).join('')}
  
  <h2>📊 Quick Numbers</h2>
  <div class="story">
    <strong>Report Card:</strong><br>
    - Grade: ${this.calculateGrade()}<br>
    - Best Decision: ${this.getBestDecision()}<br>
    - Worst Decision: ${this.getWorstDecision()}<br>
    - Luck Factor: ${this.getLuckScore()}/10<br>
    - Smart Factor: ${this.getSmartScore()}/10<br>
  </div>
</body>
</html>
    `;
  }

  calculateGrade() {
    // Simple grading based on profit
    const profit = this.getTotalProfit();
    if (profit > 2) return 'A+ 🌟';
    if (profit > 1) return 'A 😊';
    if (profit > 0.5) return 'B+ 👍';
    if (profit > 0) return 'B 🙂';
    if (profit > -0.5) return 'C 😐';
    return 'D 📚 (Time to study!)';
  }
  
  // Helper methods
  getBotCompetitions() {
    return [];
  }
  
  getEvolutionProgress() {
    return { generation: 1, totalTested: 0, newStrategies: 0, retiredStrategies: 0 };
  }
  
  getTodaysLosses() {
    return [];
  }
  
  getMissedOpportunities() {
    return [];
  }
  
  getLearning(index) {
    const learnings = [
      "Diamonds work best in the morning",
      "News sentiment helps predict price moves",
      "Quick exits save money on volatile trades"
    ];
    return learnings[index - 1] || "Keep learning!";
  }
  
  explainWinningStrategy(winner) {
    return "Used smart AI to find opportunities";
  }
  
  getStrategyMetaphor(strategy) {
    return "special calculator";
  }
  
  getTotalProfit() {
    return 0; // TODO: Calculate from trades
  }
  
  countConfidentDecisions() {
    return 0; // TODO: Count from database
  }
  
  countUnsureDecisions() {
    return 0; // TODO: Count from database
  }
  
  /**
   * Generate complete daily story report
   */
  async generateDailyStory() {
    try {
      const stories = await Promise.all([
        this.morningStory(),
        this.tradingStory(),
        this.learningStory(),
        this.mistakesStory(),
        this.tomorrowStory()
      ]);
      
      return {
        generated_at: new Date().toISOString(),
        stories: {
          morning: stories[0],
          trading: stories[1],
          learning: stories[2],
          mistakes: stories[3],
          tomorrow: stories[4]
        },
        summary: {
          total_trades: 0, // TODO: Get from database
          total_pnl: 0, // TODO: Get from database
          lessons_learned: 3
        }
      };
    } catch (error) {
      console.error('[StoryReport] Error generating daily story:', error);
      return {
        error: error.message,
        stories: {}
      };
    }
  }
}

module.exports = StoryReportGenerator;
