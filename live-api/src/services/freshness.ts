// live-api/src/services/freshness.ts
import { getQuotesCacheAgeS, getFeaturesAgeS, getWorldModelAgeS } from "./yourSources";

export function currentFreshness() {
  return {
    quotes_s: getQuotesCacheAgeS() ?? 999,
    features_s: getFeaturesAgeS() ?? 999,
    world_model_s: getWorldModelAgeS() ?? 999,
  };
}
