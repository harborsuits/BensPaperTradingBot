import type { Candidate } from "@/types/candidate";

function fmtBps(x?: number) {
  return typeof x === "number" ? `${Math.round(x)}bps` : "—";
}
function fmtPct(x?: number, digits = 0) {
  return typeof x === "number" ? `${(x*100).toFixed(digits)}%` : "—";
}
function fmtHeat(x?: number) {
  return typeof x === "number" ? `${Math.round(x*100)}%` : "—";
}
function clamp2(x: number) { return Math.round(x * 100) / 100; }

export function whyLine(c: Candidate) {
  const pieces: string[] = [];

  // headline / topic
  if (c.reason_tags?.length) {
    if (c.reason_tags.includes("tariff") && c.reason_tags.includes("semiconductors")) {
      pieces.push("Fresh tariff headlines naming semiconductors");
    } else if (c.headline) {
      pieces.push(c.headline);
    } else {
      pieces.push(c.reason_tags.join(", "));
    }
  }

  // novelty
  if (typeof c.novelty_score === "number") {
    pieces.push(`novelty ${c.novelty_score >= 0.7 ? "high" : c.novelty_score >= 0.4 ? "medium" : "low"}`);
  }

  // sentiment
  if (typeof c.sentiment_z === "number") {
    const s = c.sentiment_z;
    const sign = s > 0 ? "+" : "";
    pieces.push(`sentiment ${sign}${clamp2(s)}σ`);
  }

  // IV
  if (typeof c.iv_change_1d === "number") {
    pieces.push(c.iv_change_1d >= 0 ? "IV rising" : "IV easing");
  }

  return pieces.length ? pieces.join("; ") + "." : "Signals present.";
}

export function fitLine(c: Candidate) {
  const reg = c.regime ? c.regime.replace("_", "-") : "unknown";
  const align = c.regime_alignment ?? "neutral";
  const english =
    align === "bearish" ? "this aligns (bearish)" :
    align === "bullish" ? "this aligns (bullish)" :
    align === "neutral" ? "mixed" : "divergent";
  return `${reg} regime; ${english}.`;
}

export function costsRiskLine(c: Candidate) {
  const fees = fmtBps(c.fees_bps);
  const slip = fmtBps(c.slip_bps);
  const spread = typeof c.spread_bps === "number"
    ? `${fmtBps(c.spread_bps)}${(c.spread_cap_bps!=null && c.spread_bps > c.spread_cap_bps) ? " (wide)" : " (ok)"}`
    : "—";
  const heat = c.cluster_heat_delta != null ? fmtHeat(c.cluster_heat_delta) : "—";
  return `Fees ${fees} + slip ${slip}; spread ${spread}; cluster heat ${heat} if filled.`;
}

export function planLine(c: Candidate) {
  if (!c.plan_strategy && !c.plan_risk_usd && !c.plan_target_r && !c.plan_horizon) {
    return "No plan suggested.";
  }
  const strat =
    c.plan_strategy === "put_debit_spread" ? "tiny put debit spread" :
    c.plan_strategy === "call_debit_spread" ? "tiny call debit spread" :
    c.plan_strategy === "straight_put" ? "small straight put" :
    c.plan_strategy === "straight_call" ? "small straight call" :
    c.plan_strategy || "small position";
  const risk = c.plan_risk_usd != null ? `risk $${Math.round(c.plan_risk_usd)}` : "risk small";
  const tgt  = c.plan_target_r != null ? `target ~${c.plan_target_r}R` : undefined;
  const hz   = c.plan_horizon ? `in ${c.plan_horizon}` : undefined;
  return ["Explore:", strat, ",", risk, tgt ? `, ${tgt}` : "", hz ? `, ${hz}` : ""].join(" ").replace(/\s+/g," ").trim();
}

export function goNoGoLine(c: Candidate) {
  const meta = (c.meta_prob!=null && c.meta_threshold!=null)
    ? `meta-prob ${c.meta_prob.toFixed(2)}≥${c.meta_threshold.toFixed(2)}`
    : undefined;

  if (c.decision === "PASS" || c.decision === "PROBE") {
    const qualifier = c.decision === "PROBE" ? "→ probe mode short confirm" : "";
    return `PASS: freshness, microstructure${meta ? `, ${meta}` : ""}. ${qualifier}`.trim();
  }
  // SKIP
  let gate = "";
  if (c.skip_codes?.length) {
    gate = c.skip_codes.map(code => {
      if (code === "SPREAD_TOO_WIDE" && c.spread_bps!=null && c.spread_cap_bps!=null) {
        return `spread ${Math.round(c.spread_bps)}bps > ${Math.round(c.spread_cap_bps)}bps cap`;
      }
      if (code === "NOVELTY_LOW") return "novelty low; looks like echo";
      if (code === "META_PROB_LOW" && c.meta_prob!=null && c.meta_threshold!=null) {
        return `meta-prob ${c.meta_prob.toFixed(2)}<${c.meta_threshold.toFixed(2)}`;
      }
      return code.replace(/_/g," ").toLowerCase();
    }).join("; ");
  }
  return `SKIP: ${gate || "gate failed."}`;
}

export function buildCandidateNarrative(c: Candidate) {
  return {
    why: whyLine(c),
    fit: fitLine(c),
    costs: costsRiskLine(c),
    plan: planLine(c),
    go: goNoGoLine(c),
  };
}
