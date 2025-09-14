const express = require('express');
const router = express.Router();
const StrategyManager = require('../services/strategy_manager');
const TournamentController = require('../services/tournament_controller');
const AIOrchestrator = require('../services/ai_orchestrator');
const { MarketIndicatorsService } = require('../src/services/marketIndicators');

// Initialize services
const strategyManager = new StrategyManager();

// Mock event bus for tournament controller
const mockEventBus = {
    publish: (event) => {
        console.log('[EVENT]', event.type, event.data);
        // In production, this would emit SSE events
    }
};

const tournamentController = new TournamentController(strategyManager, mockEventBus);

// Initialize market indicators service
const marketIndicators = new MarketIndicatorsService();

// Initialize AI Orchestrator
const aiOrchestrator = new AIOrchestrator(strategyManager, tournamentController, marketIndicators);

// Start AI Orchestrator
aiOrchestrator.start();

/**
 * @route GET /api/live/status
 * @desc Get live status of WebSocket connections and configuration
 * @access Public
 */
router.get('/status', (req, res) => {
  const { wss, wssDecisions, wssPrices } = req.app.locals;

  res.json({
    prices_ws_clients: wssPrices?.clients?.size || 0,
    decisions_ws_clients: wssDecisions?.clients?.size || 0,
    quotes_refresh_ms: Number(process.env.QUOTES_REFRESH_MS || 5000),
    autorefresh: process.env.AUTOREFRESH_ENABLED === '1',
    live_quotes: process.env.QUOTES_PROVIDER !== 'synthetic' && !!process.env.TRADIER_TOKEN
  });
});

/**
 * @route GET /api/live/strategies
 * @desc Get all strategies
 * @access Public
 */
router.get('/strategies', (req, res) => {
  try {
    const strategies = strategyManager.getAllStrategies();
    res.json({ strategies });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/live/strategies/:id
 * @desc Get specific strategy
 * @access Public
 */
router.get('/strategies/:id', (req, res) => {
  try {
    const strategy = strategyManager.getStrategy(req.params.id);
    if (!strategy) {
      return res.status(404).json({ error: 'Strategy not found' });
    }
    res.json({ strategy });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/strategies
 * @desc Register new strategy
 * @access Public
 */
router.post('/strategies', (req, res) => {
  try {
    const { strategyId, config } = req.body;
    if (!strategyId || !config) {
      return res.status(400).json({ error: 'strategyId and config required' });
    }

    const strategy = strategyManager.registerStrategy(strategyId, config);
    res.json({ strategy });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route PUT /api/live/strategies/:id/performance
 * @desc Update strategy performance
 * @access Public
 */
router.put('/strategies/:id/performance', (req, res) => {
  try {
    const { metrics } = req.body;
    if (!metrics) {
      return res.status(400).json({ error: 'metrics required' });
    }

    const success = strategyManager.updatePerformance(req.params.id, metrics);
    if (!success) {
      return res.status(404).json({ error: 'Strategy not found' });
    }

    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/strategies/:id/promote
 * @desc Promote strategy from paper to live
 * @access Public
 */
router.post('/strategies/:id/promote', (req, res) => {
  try {
    const { reason = 'manual_promotion' } = req.body;
    const result = strategyManager.manualPromotion(req.params.id, reason);

    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/strategies/:id/demote
 * @desc Demote strategy from live to paper
 * @access Public
 */
router.post('/strategies/:id/demote', (req, res) => {
  try {
    const { reason = 'manual_demotion' } = req.body;
    const result = strategyManager.demoteStrategy(req.params.id, reason);

    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/live/promotions
 * @desc Get promotion history
 * @access Public
 */
router.get('/promotions', (req, res) => {
  try {
    const { strategyId } = req.query;
    const promotions = strategyManager.getPromotionHistory(strategyId);
    res.json({ promotions });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/live/promotion-candidates
 * @desc Get strategies ready for promotion
 * @access Public
 */
router.get('/promotion-candidates', (req, res) => {
  try {
    const candidates = strategyManager.getPromotionCandidates();
    res.json({ candidates });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/live/dashboard
 * @desc Get strategy management dashboard data
 * @access Public
 */
router.get('/dashboard', (req, res) => {
  try {
    const dashboard = strategyManager.getDashboardData();
    res.json(dashboard);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/live/tournament
 * @desc Get tournament dashboard data
 * @access Public
 */
router.get('/tournament', (req, res) => {
  try {
    const tournamentData = tournamentController.getDashboardData();
    res.json(tournamentData);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/tournament/register-phenotype
 * @desc Register new phenotype from EvoTester
 * @access Public
 */
router.post('/tournament/register-phenotype', (req, res) => {
  try {
    const { phenotypeData } = req.body;
    if (!phenotypeData) {
      return res.status(400).json({ error: 'phenotypeData required' });
    }

    const strategy = tournamentController.registerPhenotype(phenotypeData);
    res.json({ strategy });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/tournament/force-decision
 * @desc Force tournament decision for testing
 * @access Public
 */
router.post('/tournament/force-decision', (req, res) => {
  try {
    const { strategyId, decision, reason = 'manual_override' } = req.body;

    if (!strategyId || !decision) {
      return res.status(400).json({ error: 'strategyId and decision required' });
    }

    let result;
    if (decision === 'promote') {
      result = strategyManager.manualPromotion(strategyId, reason);
    } else if (decision === 'demote') {
      result = strategyManager.demoteStrategy(strategyId, reason);
    } else {
      return res.status(400).json({ error: 'Invalid decision. Use "promote" or "demote"' });
    }

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/ai/manage-strategies
 * @desc AI-driven strategy management (called by brain system)
 * @access Public
 */
router.post('/ai/manage-strategies', (req, res) => {
  try {
    const { currentMetrics = {} } = req.body;
    const results = strategyManager.evaluateAndManageStrategies(currentMetrics);

    console.log(`[AI] Strategy management completed:`, results);

    res.json({
      success: true,
      results: results,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error(`[AI] Strategy management error:`, error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/ai/register-strategy
 * @desc AI registers a new strategy automatically
 * @access Public
 */
router.post('/ai/register-strategy', (req, res) => {
  try {
    const { strategyId, config } = req.body;

    if (!strategyId || !config) {
      return res.status(400).json({ error: 'strategyId and config required' });
    }

    const strategy = strategyManager.registerStrategy(strategyId, config);

    console.log(`[AI] Auto-registered strategy: ${strategyId}`);

    res.json({
      success: true,
      strategy: strategy
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/live/ai/update-performance
 * @desc AI updates strategy performance metrics
 * @access Public
 */
router.post('/ai/update-performance', (req, res) => {
  try {
    const { strategyId, metrics } = req.body;

    if (!strategyId || !metrics) {
      return res.status(400).json({ error: 'strategyId and metrics required' });
    }

    const success = strategyManager.updatePerformance(strategyId, metrics);

    if (!success) {
      return res.status(404).json({ error: 'Strategy not found' });
    }

    console.log(`[AI] Updated performance for ${strategyId}`);

    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/ai/policy
 * @desc Get current AI policy configuration
 * @access Public
 */
router.get('/ai/policy', (req, res) => {
  try {
    const policy = aiOrchestrator.policy.ai_policy || {};
    res.json({
      paper_cap_max: policy.paper_cap_max || 20000,
      r1_max: policy.rounds?.R1?.max_slots || 50,
      r2_max: policy.rounds?.R2?.max_slots || 20,
      r3_max: policy.rounds?.R3?.max_slots || 8,
      exploration_quota: policy.exploration_quota || 0.1,
      families: aiOrchestrator.policy.families || {},
      triggers: aiOrchestrator.policy.triggers || {},
      guardrails: aiOrchestrator.policy.guardrails || {}
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/ai/context
 * @desc Get current market context and system state
 * @access Public
 */
router.get('/ai/context', (req, res) => {
  try {
    const context = aiOrchestrator.marketContext || {};
    const roster = aiOrchestrator.getRosterSnapshot();
    const capacity = aiOrchestrator.getCapacitySnapshot();

    res.json({
      regime: context.regime || 'neutral',
      volatility: context.volatility || 'medium',
      sentiment: context.sentiment || 'neutral',
      vix_level: context.vix_level,
      calendar_events: context.calendar_events || [],
      roster_metrics: {
        total_strategies: roster.total,
        by_stage: roster.byStage,
        by_status: roster.byStatus,
        avg_sharpe: roster.performance.avgSharpe,
        avg_pf: roster.performance.avgPF,
        underperformers: roster.performance.underperformers
      },
      capacity: {
        paper_budget_used: capacity.paperBudget.used,
        paper_budget_max: capacity.paperBudget.max,
        paper_budget_available: capacity.paperBudget.available,
        slots_r1_available: capacity.slots.R1.available,
        slots_r2_available: capacity.slots.R2.available,
        slots_r3_available: capacity.slots.R3.available
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/ai/evo/seed
 * @desc Request EvoTester to generate new phenotypes
 * @access Public
 */
router.post('/ai/evo/seed', (req, res) => {
  try {
    const { families, count, bounds, objective } = req.body;

    if (!families || !Array.isArray(families) || families.length === 0) {
      return res.status(400).json({ error: 'families array required' });
    }

    if (!count || count <= 0) {
      return res.status(400).json({ error: 'valid count required' });
    }

    // Use AI orchestrator's seed generator
    const phenotypes = aiOrchestrator.seedGenerator.generatePhenotypes({
      families,
      count,
      bounds: bounds || {},
      objective: objective || 'after_cost_sharpe_pf_dd'
    });

    console.log(`[AI] EvoTester seeding: ${count} phenotypes requested, ${phenotypes.length} generated`);

    res.json({
      success: true,
      phenotypes: phenotypes,
      count_generated: phenotypes.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/ai/evo/feedback
 * @desc Send promotion/demotion feedback to EvoTester
 * @access Public
 */
router.post('/ai/evo/feedback', (req, res) => {
  try {
    const { generation, results } = req.body;

    if (!results || !Array.isArray(results)) {
      return res.status(400).json({ error: 'results array required' });
    }

    // Process feedback for learning
    const feedback = {
      generation: generation || aiOrchestrator.tournamentController.currentGeneration,
      results: results,
      timestamp: new Date().toISOString()
    };

    // In production, this would POST to EvoTester's feedback endpoint
    console.log(`[AI] EvoTester feedback: ${results.length} results for generation ${feedback.generation}`);

    // Store feedback for analysis
    aiOrchestrator.processEvoFeedback?.(feedback) || console.log('Feedback processed:', feedback);

    res.json({
      success: true,
      feedback_processed: results.length,
      generation: feedback.generation,
      timestamp: feedback.timestamp
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/ai/status
 * @desc Get AI orchestrator status and recent activity
 * @access Public
 */
router.get('/ai/status', (req, res) => {
  try {
    const status = aiOrchestrator.getStatus();

    res.json({
      is_active: status.isActive,
      last_run: status.lastRun,
      total_cycles: status.totalCycles,
      current_regime: status.currentContext?.regime,
      recent_decisions: status.recentDecisions?.slice(-3) || [],
      policy_version: status.policyVersion,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/ai/trigger-cycle
 * @desc Manually trigger an AI orchestration cycle
 * @access Public
 */
router.post('/ai/trigger-cycle', (req, res) => {
  try {
    aiOrchestrator.triggerManualCycle();

    res.json({
      success: true,
      message: 'AI orchestration cycle triggered',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/ai/decision-history
 * @desc Get recent AI decision history
 * @access Public
 */
router.get('/ai/decision-history', (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const history = aiOrchestrator.decisionHistory?.slice(-limit) || [];

    res.json({
      decisions: history,
      total: history.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/live/ai/status
 * @desc Get AI orchestrator status
 * @access Public
 */
router.get('/ai/status', (req, res) => {
  try {
    const aiStatus = {
      is_active: true,
      last_run: new Date().toISOString(),
      total_cycles: 0, // Would come from AI orchestrator
      current_regime: 'neutral_medium', // From market context
      recent_decisions: [],
      policy_version: 'latest',
      timestamp: new Date().toISOString(),
      circuit_breakers: []
    };
    res.json(aiStatus);
  } catch (error) {
    console.error('AI status error:', error);
    res.status(500).json({ error: 'Failed to get AI status' });
  }
});

/**
 * @route GET /api/live/ai/context
 * @desc Get AI market context and roster information
 * @access Public
 */
router.get('/ai/context', (req, res) => {
  try {
    // Get real roster data from AI orchestrator
    const roster = aiOrchestrator.getRosterSnapshot();

    const aiContext = {
      regime: 'neutral_medium',
      volatility: 'medium',
      sentiment: 'neutral',
      vix_level: 18.5,
      calendar_events: [],
      roster_metrics: {
        total_strategies: roster.total,
        by_stage: roster.byStage,
        by_status: roster.byStatus,
        avg_sharpe: roster.performance.avgSharpe,
        avg_pf: roster.performance.avgPF,
        underperformers: roster.performance.underperformers
      },
      timestamp: new Date().toISOString()
    };

    res.json(aiContext);
  } catch (error) {
    console.error('AI context error:', error);
    res.status(500).json({ error: 'Failed to get AI context' });
  }
});

module.exports = router;
