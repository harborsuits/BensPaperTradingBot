// live-api/src/services/BrainService.ts
import axios from "axios";

const PY_BRAIN = process.env.PY_BRAIN_URL || "http://localhost:8001";
const TIMEOUT_MS = +(process.env.BRAIN_TIMEOUT_MS || 450);

type ScoreResp = {
  final_score: number;
  conf: number;
  experts: Array<{ name: string; score: number; conf: number; why?: string[]; weight?: number }>;
  world_model: Record<string, any>;
  policy_used_id: string;
};

export async function scoreSymbol(symbol: string, snapshot_ts?: string) {
  try {
    // Use the correct Python endpoint: /api/decide
    const { data } = await axios.post(
      `${PY_BRAIN}/api/decide`,
      {
        opportunities: [{
          symbol,
          alpha: 0.5, // Will be computed by Python brain
          sentiment_boost: 0,
          regime_align: 0.5,
          est_cost_bps: 20,
          risk_penalty: 0
        }],
        current_regime: 'neutral'
      },
      { timeout: TIMEOUT_MS }
    );

    // Transform Python response to Node format
    if (data.decisions && data.decisions[0]) {
      const decision = data.decisions[0];
      return {
        symbol: decision.symbol,
        experts: [
          { name: "technical", score: decision.confidence || 0.5, conf: 0.8 },
          { name: "news", score: decision.confidence || 0.5, conf: 0.7 },
          { name: "volatility", score: decision.confidence || 0.5, conf: 0.6 },
          { name: "statistical", score: decision.confidence || 0.5, conf: 0.75 }
        ],
        final_score: decision.confidence || 0.5,
        conf: decision.confidence || 0.5,
        world_model: {
          regime: 'neutral',
          volatility: 'medium',
          trend: 'up'
        },
        policy_used_id: "python_brain",
        snapshot_ts: snapshot_ts || "NOW",
        processing_ms: data.processing_time_ms || 50,
        fallback: false
      };
    }

    throw new Error("No decision returned from Python brain");
  } catch (err) {
    // Surface fallback so UI can show "Degraded (ML offline)"
    return {
      final_score: 0,
      conf: 0,
      experts: [],
      world_model: {},
      policy_used_id: "fallback-sim",
      fallback: true,
      error: (err as Error)?.message || "python_brain_unreachable",
    };
  }
}

type PlanReq = { symbol: string; final_score: number; conf: number; account_state?: any };
type PlanResp = { action: "enter"|"hold"|"exit"|"avoid"|"halt"; sizing_dollars?: number; stop?: number; target?: number; structure?: any; reason?: string };

export async function planTrade(req: PlanReq) {
  try {
    // For now, use simple rule-based planning since Python doesn't have plan endpoint
    // TODO: Add /api/brain/plan to Python service
    const { symbol, final_score, conf, account_state } = req;

    let action: "enter"|"hold"|"exit"|"avoid"|"halt" = "hold";
    let sizing_dollars = 0;
    let stop = null;
    let target = null;
    let structure = null;
    let reason = "python_brain_plan";

    if (final_score > 0.8 && conf > 0.7) {
      action = "enter";
      sizing_dollars = Math.min(account_state?.buying_power || 10000, 5000);
      reason = "high_confidence_entry";
    } else if (final_score < 0.3) {
      action = "avoid";
      reason = "low_confidence_avoid";
    }

    return {
      action,
      sizing_dollars,
      stop,
      target,
      structure,
      reason,
      fallback: false
    };
  } catch (err) {
    return { action: "avoid", reason: "fallback_brain_unreachable", fallback: true };
  }
}
