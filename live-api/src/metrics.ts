// live-api/src/metrics.ts
import client from "prom-client";
export const register = new client.Registry();
const scoreLatency = new client.Histogram({ name:"brain_score_latency_ms", help:"score latency", buckets:[25,50,100,200,300,500,800,1200] });
const planLatency  = new client.Histogram({ name:"brain_plan_latency_ms", help:"plan latency",  buckets:[25,50,100,200,300,500,800,1200] });
const quotesFresh  = new client.Gauge({ name:"fresh_quotes_s", help:"quotes age seconds" });
const featsFresh   = new client.Gauge({ name:"fresh_features_s", help:"features age seconds" });
const wmFresh      = new client.Gauge({ name:"fresh_world_model_s", help:"world model age seconds" });
register.registerMetric(scoreLatency); register.registerMetric(planLatency);
register.registerMetric(quotesFresh); register.registerMetric(featsFresh); register.registerMetric(wmFresh);

export function observeFreshness(f:{quotes_s:number;features_s:number;world_model_s:number}) {
  quotesFresh.set(f.quotes_s); featsFresh.set(f.features_s); wmFresh.set(f.world_model_s);
}
