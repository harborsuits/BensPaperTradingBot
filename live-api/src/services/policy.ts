// live-api/src/services/policy.ts
import dayjs from "dayjs";

export async function withinEarningsWindow(symbol: string) {
  // Replace with your calendar source; placeholder returns null or { minutes_to_event: number }
  const info = await getEarningsInfo(symbol); // implement or call your existing helper
  if (!info) return false;
  const mins = info.minutes_to_event ?? 9999;
  const window = +(process.env.EARNINGS_FORBID_MIN || 60);
  return Math.abs(mins) <= window; // within ±window
}
