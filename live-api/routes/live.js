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

/**
 * @route GET /api/decisions/recent
 * @desc Get recent decisions by stage (proposed, intent, executed)
 * @access Public
 * @query {string} stage - The stage to filter by (proposed|intent|executed), defaults to 'proposed'
 * @query {number} limit - Maximum number of items to return, defaults to 50
 */
router.get('/decisions/recent', (req, res) => {
  try {
    const stage = req.query.stage || 'proposed';
    const limit = parseInt(req.query.limit) || 50;

    // Validate stage parameter
    const validStages = ['proposed', 'intent', 'executed'];
    if (!validStages.includes(stage)) {
      return res.status(400).json({ error: `Invalid stage. Must be one of: ${validStages.join(', ')}` });
    }

    // Get decisions from the decision ring buffer (simulated for now)
    // In production, this would be a persistent ring buffer in memory
    const decisions = getRecentDecisionsByStage(stage, limit);

    res.json(decisions);
  } catch (error) {
    console.error('Decisions recent error:', error);
    res.status(500).json({ error: 'Failed to get recent decisions' });
  }
});

/**
 * @route GET /api/brain/status
 * @desc Get brain loop status and recent metrics
 * @access Public
 */
router.get('/brain/status', (req, res) => {
  try {
    const brainStatus = {
      mode: process.env.AUTOLOOP_MODE || 'discovery',
      running: true, // Would come from actual loop status
      tick_ms: 30000,
      breaker: null,
      recent_pf_after_costs: 1.05,
      sharpe_30d: 0.42,
      sharpe_90d: 0.38,
      timestamp: new Date().toISOString()
    };

    res.json(brainStatus);
  } catch (error) {
    console.error('Brain status error:', error);
    res.status(500).json({ error: 'Failed to get brain status' });
  }
});

/**
 * @route GET /api/brain/flow/recent
 * @desc Get recent brain flow ticks per symbol
 * @access Public
 * @query {number} limit - Maximum number of ticks to return, defaults to 100
 */
router.get('/brain/flow/recent', (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;

    // Simulated brain flow data - in production this would be from the actual loop
    const brainFlowData = getRecentBrainFlowTicks(limit);

    res.json(brainFlowData);
  } catch (error) {
    console.error('Brain flow recent error:', error);
    res.status(500).json({ error: 'Failed to get brain flow data' });
  }
});

/**
 * @route GET /api/brain/scoring/activity
 * @desc Get scoring activity for a specific symbol and tick
 * @access Public
 * @query {string} symbol - The symbol to get scoring for
 * @query {string} ts - The timestamp of the tick (optional, defaults to latest)
 */
router.get('/brain/scoring/activity', (req, res) => {
  try {
    const { symbol, ts } = req.query;

    if (!symbol) {
      return res.status(400).json({ error: 'symbol parameter is required' });
    }

    // Get scoring activity for the symbol
    const scoringActivity = getScoringActivityForSymbol(symbol, ts);

    if (!scoringActivity) {
      return res.status(404).json({ error: 'No scoring activity found for the specified symbol and timestamp' });
    }

    res.json(scoringActivity);
  } catch (error) {
    console.error('Brain scoring activity error:', error);
    res.status(500).json({ error: 'Failed to get scoring activity' });
  }
});

/**
 * @route GET /api/evo/status
 * @desc Get EvoTester status and recent activity
 * @access Public
 */
router.get('/evo/status', (req, res) => {
  try {
    const evoStatus = {
      generation: 15,
      population: 200,
      best: {
        config_id: 'cfg_abc123',
        metrics: {
          pf_after_costs: 1.18,
          sharpe: 0.42,
          trades: 640
        }
      },
      running: true,
      timestamp: new Date().toISOString()
    };

    res.json(evoStatus);
  } catch (error) {
    console.error('Evo status error:', error);
    res.status(500).json({ error: 'Failed to get evo status' });
  }
});

/**
 * @route GET /api/evo/candidates
 * @desc Get recent EvoTester candidates ready for promotion
 * @access Public
 * @query {number} limit - Maximum number of candidates to return, defaults to 20
 */
router.get('/evo/candidates', (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;

    // Get recent candidates from EvoTester
    const candidates = getRecentEvoCandidates(limit);

    res.json(candidates);
  } catch (error) {
    console.error('Evo candidates error:', error);
    res.status(500).json({ error: 'Failed to get evo candidates' });
  }
});

/**
 * @route POST /api/evo/schedule-paper-validate
 * @desc Schedule paper validation for a candidate config
 * @access Public
 * @body {string} config_id - The configuration ID to validate
 * @body {number} days - Number of days to run validation (default: 14)
 */
router.post('/evo/schedule-paper-validate', (req, res) => {
  try {
    const { config_id, days = 14 } = req.body;

    if (!config_id) {
      return res.status(400).json({ error: 'config_id is required' });
    }

    // Schedule paper validation
    const trackingId = schedulePaperValidation(config_id, days);

    res.json({
      success: true,
      tracking_id: trackingId,
      config_id,
      days,
      message: `Paper validation scheduled for ${days} days`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Evo schedule paper validate error:', error);
    res.status(500).json({ error: 'Failed to schedule paper validation' });
  }
});

/**
 * @route GET /api/evo/validation/:id
 * @desc Get paper validation progress for a scheduled validation
 * @access Public
 * @param {string} id - The tracking ID of the validation
 */
router.get('/evo/validation/:id', (req, res) => {
  try {
    const { id } = req.params;

    // Get validation progress
    const validation = getValidationProgress(id);

    if (!validation) {
      return res.status(404).json({ error: 'Validation not found' });
    }

    res.json(validation);
  } catch (error) {
    console.error('Evo validation progress error:', error);
    res.status(500).json({ error: 'Failed to get validation progress' });
  }
});

/**
 * @route GET /api/evo/validation/:id/gates
 * @desc Get gate evaluation results for a validation
 * @access Public
 * @param {string} id - The tracking ID of the validation
 */
router.get('/evo/validation/:id/gates', (req, res) => {
  try {
    const { id } = req.params;

    // Get gate evaluation results
    const gates = getValidationGates(id);

    if (!gates) {
      return res.status(404).json({ error: 'Validation gates not found' });
    }

    res.json(gates);
  } catch (error) {
    console.error('Evo validation gates error:', error);
    res.status(500).json({ error: 'Failed to get validation gates' });
  }
});

/**
 * @route GET /api/audit/autoloop/status
 * @desc Get current autoloop status (for proof commands)
 * @access Public
 */
router.get('/audit/autoloop/status', (req, res) => {
  try {
    // Read from evidence file or get from actual status
    const status = {
      mode: process.env.AUTOLOOP_MODE || 'discovery',
      status: 'running',
      running: true,
      timestamp: new Date().toISOString()
    };

    res.json(status);
  } catch (error) {
    console.error('Audit autoloop status error:', error);
    res.status(500).json({ error: 'Failed to get autoloop status' });
  }
});

/**
 * @route GET /api/brain/flow/summary
 * @desc Get brain flow summary for dashboard (counts by stage/mode + latency)
 * @access Public
 * @query {string} window - Time window (15m, 1h, 1d), defaults to 15m
 */
router.get('/brain/flow/summary', (req, res) => {
  try {
    const window = req.query.window || '15m';

    // Simulated summary data - in production this would be aggregated from actual flow data
    const summary = {
      window,
      counts: {
        ingest_ok: 87,
        context_ok: 85,
        candidates_ok: 82,
        gates_passed: 41,
        gates_failed: 46,
        plan_ok: 2, // Low in discovery mode
        route_ok: 1,
        manage_ok: 1,
        learn_ok: 1
      },
      by_mode: {
        discovery: 100,
        shadow: 0,
        live: 0
      },
      latency_ms: {
        p50: 120,
        p95: 340
      },
      timestamp: new Date().toISOString()
    };

    res.json(summary);
  } catch (error) {
    console.error('Brain flow summary error:', error);
    res.status(500).json({ error: 'Failed to get brain flow summary' });
  }
});

/**
 * @route GET /api/decisions/summary
 * @desc Get decisions summary for dashboard (proposals/min, unique symbols, last ts)
 * @access Public
 * @query {string} window - Time window (15m, 1h, 1d), defaults to 15m
 */
router.get('/decisions/summary', (req, res) => {
  try {
    const window = req.query.window || '15m';

    // Simulated summary data - in production this would be aggregated from actual decisions
    const summary = {
      window,
      proposals_per_min: 4.2,
      unique_symbols: 7,
      last_ts: new Date().toISOString(),
      by_stage: {
        proposed: 15,
        intent: 2,
        executed: 8
      },
      timestamp: new Date().toISOString()
    };

    res.json(summary);
  } catch (error) {
    console.error('Decisions summary error:', error);
    res.status(500).json({ error: 'Failed to get decisions summary' });
  }
});

// Helper functions for data simulation (would be replaced with real implementations)
function getRecentDecisionsByStage(stage, limit) {
  // Simulated data - in production this would be from persistent ring buffers
  const now = new Date();

  if (stage === 'proposed') {
    return [
      {
        id: `dec_${Date.now()}`,
        ts: now.toISOString(),
        symbol: 'SPY',
        strategy_id: 'news_momo_v2',
        confidence: 0.71,
        costs_est_bps: 8,
        winner_score: 0.63,
        reason: 'news impulse + momo filter',
        mode: 'discovery',
        trace_id: `trace_${Date.now()}`
      }
    ].slice(0, limit);
  } else if (stage === 'intent') {
    return [
      {
        id: `ti_${Date.now()}`,
        ts: now.toISOString(),
        symbol: 'SPY',
        side: 'BUY',
        qty: 5,
        limit: 443.12,
        tif: 'DAY',
        strategy_id: 'news_momo_v2',
        confidence: 0.71,
        ev_after_costs: 0.0032,
        risk_summary: {
          gates: ['hours_ok', 'spread_ok']
        },
        mode: 'shadow',
        trace_id: `trace_${Date.now()}`
      }
    ].slice(0, limit);
  } else if (stage === 'executed') {
    // Return paper orders as executions
    return [
      {
        id: `exec_${Date.now()}`,
        ts: now.toISOString(),
        symbol: 'SPY',
        side: 'BUY',
        qty: 5,
        price: 443.12,
        status: 'FILLED',
        strategy_id: 'news_momo_v2',
        order_type: 'LIMIT',
        tif: 'DAY'
      }
    ].slice(0, limit);
  }

  return [];
}

function getRecentBrainFlowTicks(limit) {
  // Simulated brain flow data
  const now = new Date();

  return [
    {
      symbol: 'SPY',
      ts: now.toISOString(),
      stages: {
        ingest: { ok: true, quote_age_s: 1.2 },
        context: { ok: true, vol_rank: 0.42, atr: 3.1 },
        candidates: { ok: true, count: 4, winner: { strategy_id: 'news_momo_v2', confidence: 0.71 } },
        gates: { ok: true, passed: ['hours_ok', 'spread_ok'], rejected: [] },
        plan: { ok: false, reason: 'discovery_mode' },
        route: { ok: false, skipped: true },
        manage: { ok: false, skipped: true },
        learn: { ok: false, skipped: true }
      },
      mode: 'discovery',
      trace_id: `trace_${Date.now()}`
    }
  ].slice(0, limit);
}

function getScoringActivityForSymbol(symbol, ts) {
  // Simulated scoring activity data
  const now = new Date();

  return {
    symbol: symbol,
    ts: ts || now.toISOString(),
    candidates: [
      {
        strategy_id: 'news_momo_v2',
        raw_score: 0.82,
        ev_after_costs: 0.0032,
        reliability: 0.78,
        liquidity: 0.93,
        total: 0.82 * 0.78 * 0.93,
        selected: true,
        reason: 'momo+news; spread 6bps; PF 1.14 (90d); COST_OK'
      },
      {
        strategy_id: 'mean_rev',
        raw_score: 0.49,
        ev_after_costs: -0.0004,
        reliability: 0.55,
        liquidity: 0.98,
        total: -0.00021,
        selected: false,
        reason: 'ev<=0 after costs (blocked)'
      }
    ],
    weights: { ev: 1.0, reliability: 1.0, liquidity: 1.0 },
    trace_id: `trace_${Date.now()}`
  };
}

function getRecentEvoCandidates(limit) {
  // Simulated EvoTester candidates
  return [
    {
      config_id: 'cfg_abc123',
      strategy_id: 'news_momo_v2',
      params: { lookback: 25, z_entry: 1.8 },
      backtest: {
        pf_after_costs: 1.18,
        sharpe: 0.42,
        trades: 640,
        regimes: { volatile: 1.14, quiet: 1.07 }
      },
      ready_for_paper: true
    }
  ].slice(0, limit);
}

function schedulePaperValidation(configId, days) {
  // Simulate scheduling paper validation
  const trackingId = `val_${Date.now()}`;
  console.log(`[EVO] Scheduled paper validation for ${configId} (${days} days), tracking: ${trackingId}`);
  return trackingId;
}

function getValidationProgress(id) {
  // Simulate validation progress
  return {
    tracking_id: id,
    config_id: 'cfg_abc123',
    status: 'running',
    days_completed: 7,
    total_days: 14,
    current_pf: 1.08,
    current_sharpe: 0.38,
    timestamp: new Date().toISOString()
  };
}

function getValidationGates(id) {
  // Simulate gate evaluation
  return {
    passed: true,
    details: {
      pf_gate: { required: 1.05, actual: 1.08, passed: true },
      sharpe_gate: { required: 0.3, actual: 0.38, passed: true },
      trades_gate: { required: 100, actual: 320, passed: true },
      regime_gate: { required: ['volatile', 'quiet'], actual: ['volatile', 'quiet'], passed: true }
    }
  };
}

module.exports = router;
