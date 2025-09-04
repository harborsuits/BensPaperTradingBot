// src/adapters/decisionTraceToEvidence.ts
import { DecisionTrace } from "@/types/DecisionTrace";
import type { EvidencePacket, SourceDoc, GateCheck } from "@/contracts/evidence";

/**
 * Converts a DecisionTrace to an EvidencePacket for compatibility with the existing EvidenceDrawer
 */
export function toEvidencePacket(d: DecisionTrace): EvidencePacket {
  // Map news evidence to source docs
  const sources: SourceDoc[] = d.news_evidence.map((n, i) => ({
    id: `${d.trace_id}-src-${i}`,
    url: n.url,
    title: n.headline,
    publisher: new URL(n.url).hostname.replace(/^www\./, ''),
    publishedAt: d.as_of, // Use decision timestamp as fallback
    capturedAt: d.as_of,
    biasLean: "unknown", // We don't have this info in DecisionTrace
    credibility: mapCredibility(n.credibility),
    sentiment: n.sentiment ? parseFloat(mapSentiment(n.sentiment)) : 0,
    relevance: 0.8, // Default high relevance
    keyClaims: [n.snippet],
  }));

  // Map risk gates
  const gates: GateCheck[] = [];
  if (d.risk_gate) {
    if (d.risk_gate.position_limits_ok !== undefined) {
      gates.push({
        name: "position_limits_ok",
        passed: !!d.risk_gate.position_limits_ok,
        details: d.risk_gate.position_limits_ok ? "Position limits OK" : "Position limits exceeded",
      });
    }
    if (d.risk_gate.portfolio_heat_ok !== undefined) {
      gates.push({
        name: "portfolio_heat_ok",
        passed: !!d.risk_gate.portfolio_heat_ok,
        details: d.risk_gate.portfolio_heat_ok ? "Portfolio heat OK" : "Portfolio heat too high",
      });
    }
    if (d.risk_gate.drawdown_ok !== undefined) {
      gates.push({
        name: "drawdown_ok",
        passed: !!d.risk_gate.drawdown_ok,
        details: d.risk_gate.drawdown_ok ? "Drawdown OK" : "Drawdown limit exceeded",
      });
    }
  }

  // Build the evidence packet
  return {
    id: d.trace_id,
    createdAt: d.as_of,
    strategyId: d.plan.strategyLabel || "unknown",
    strategyName: d.plan.strategyLabel || "Unknown Strategy",
    symbol: d.symbol,
    context: {
      regime: d.market_context?.regime?.label || "Unknown",
      vix: d.market_context?.volatility?.vix,
      marketBias: mapSentiment(d.market_context?.sentiment?.label || "neutral"),
    },
    tlDr: d.explain_layman,
    confidence: d.candidate_score?.alpha || 0.5,
    sources,
    crossConfirmations: new Set(sources.map(s => s.publisher)).size,
    interpretation: d.explain_detail,
    prediction: {
      thesis: d.explain_layman,
      direction: mapDirection(d.plan.action),
      horizonHours: 48, // Default if not specified
      expectedMovePct: d.plan.expected_move?.pct || 0,
      prob: d.candidate_score?.alpha || 0.5,
      bandsPct: {
        p10: -3, // Defaults
        p50: 4,
        p90: 9,
      },
      invalidation: "Close below stop or news reversal",
    },
    plan: {
      strategyLabel: d.plan.strategyLabel || "Strategy",
      params: {
        ...d.plan.entry,
        ...d.plan.sizing,
        ...d.plan.exits,
      },
      alternatives: [],
    },
    risk: {
      maxLossUsd: d.plan.sizing?.max_loss || 0,
      maxPortfolioHeatAfter: 0.02, // Default
      stopPlan: String(d.plan.exits?.stop || "ATR×1.2, time-box 2 days"),
      rollOrHedgeRules: ["tighten if IV expands 20%+"],
    },
    gates,
    orders: [{
      venue: d.account?.mode || "paper",
      instrument: d.instrument?.type || "equity",
      symbol: d.symbol,
      quantity: d.plan.sizing?.units || 1,
      timeInForce: "day",
      preTrade: {
        mid: d.plan.entry?.px || 0,
        spreadPct: 0.5, // Default
        iv: undefined,
      },
      idempotencyKey: `${d.trace_id}-0`,
    }],
    execution: {
      status: mapExecutionStatus(d.execution.status),
      brokerOrderIds: d.execution.broker_ids,
    },
    monitoring: [],
  };
}

// Helper functions to map between formats
function mapCredibility(cred?: string): 1 | 2 | 3 | 4 | 5 {
  if (!cred) return 3;
  switch (cred) {
    case "high": return 5;
    case "medium": return 3;
    case "low": return 1;
    default: return 3;
  }
}

function mapSentiment(sentiment?: string): "bull" | "bear" | "neutral" | "volatile" {
  if (!sentiment) return "neutral";
  switch (sentiment) {
    case "positive":
    case "bullish": return "bull";
    case "negative":
    case "bearish": return "bear";
    case "mixed": return "volatile";
    default: return "neutral";
  }
}

function mapDirection(action?: string): "up" | "down" | "volatile" {
  if (!action) return "volatile";
  switch (action) {
    case "OPEN_LONG": return "up";
    case "OPEN_SHORT": return "down";
    default: return "volatile";
  }
}

function mapExecutionStatus(status?: string): "pending" | "live" | "filled" | "rejected" | "cancelled" | "closed" {
  if (!status) return "pending";
  switch (status) {
    case "SENT": return "live";
    case "FILLED": return "filled";
    case "PARTIAL": return "live";
    case "BLOCKED": return "rejected";
    case "CANCELED": return "cancelled";
    default: return "pending";
  }
}

