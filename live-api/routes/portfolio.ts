import { Router } from "express";
import { z } from "zod";
const r = Router();

// Portfolio schemas for validation
const Position = z.object({
  symbol: z.string(),
  qty: z.number(),
  mark: z.number().nullable().default(0),
  side: z.enum(["LONG", "SHORT"]).default("LONG"),
});

const PortfolioSummaryResponse = z.object({
  broker: z.string(),
  mode: z.string(),
  asOf: z.string(),
  cash: z.number(),
  equity: z.number(),
  day_pnl: z.number(),
  open_pnl: z.number(),
  positions: z.array(Position),
});

type PositionType = z.infer<typeof Position>;
type PortfolioSummaryType = z.infer<typeof PortfolioSummaryResponse>;

// Import the response helper we created earlier
import { sendJson } from "../api/contracts/respond";

// Import the real TradierBroker for paper trading
const { TradierBroker } = require('../lib/tradierBroker');

// Real Tradier paper account adapter
const tradierPaperAccount = {
  async get() {
    try {
      const broker = new TradierBroker();
      const portfolio = await broker.getPortfolio();

      // Transform Tradier format to our expected format
      return {
        total_cash: portfolio.cash,
        cash: portfolio.cash,
        long_market_value: portfolio.equity, // Approximate - Tradier doesn't separate long/short MV
        short_market_value: 0, // Would need to calculate from positions
        day_pnl: portfolio.day_pnl,
        open_pnl: portfolio.open_pnl,
        positions: portfolio.positions.map(pos => ({
          symbol: pos.symbol,
          qty: pos.qty,
          price: pos.avg_cost, // Use average cost as price
          mark: pos.last, // Use last price as mark
          side: pos.qty >= 0 ? "LONG" : "SHORT"
        }))
      };
    } catch (error) {
      console.error('Tradier paper account error:', error);
      // Fallback to safe defaults if Tradier is unavailable
      return {
        total_cash: 0,
        cash: 0,
        long_market_value: 0,
        short_market_value: 0,
        day_pnl: 0,
        open_pnl: 0,
        positions: []
      };
    }
  }
};

r.get("/summary", async (_req, res) => {
  try {
    // Pull from your real Tradier paper account adapter
    const acct = await tradierPaperAccount.get();

    // Parse and validate positions with fallback to empty array
    const rawPositions = acct.positions ?? [];
    const positions: PositionType[] = rawPositions.map((p: any) => {
      const parsed = Position.safeParse({
        symbol: p.symbol,
        qty: Number(p.qty ?? 0),
        mark: Number(p.mark ?? p.price ?? 0),
        side: (p.side ?? (p.qty >= 0 ? "LONG" : "SHORT")).toUpperCase(),
      });

      if (!parsed.success) {
        console.warn('Invalid position data:', p, parsed.error);
        return null;
      }
      return parsed.data;
    }).filter(Boolean) as PositionType[];

    // Calculate market values server-side (single source of truth)
    const longPositions = positions.filter(p => p.side === "LONG");
    const shortPositions = positions.filter(p => p.side === "SHORT");

    const longMV = longPositions.reduce((sum, p) => sum + p.qty * (p.mark ?? 0), 0);
    const shortMV = shortPositions.reduce((sum, p) => sum + Math.abs(p.qty) * (p.mark ?? 0), 0);

    // Server is authoritative for equity calculation
    const cash = Number(acct.total_cash ?? acct.cash ?? 0);
    const equity = cash + longMV - shortMV;
    const day_pnl = Number(acct.day_pnl ?? 0);
    const open_pnl = Number(acct.open_pnl ?? 0);

    const summaryData: PortfolioSummaryType = {
      broker: "tradier",
      mode: "paper",
      asOf: new Date().toISOString(),
      cash,
      equity,
      day_pnl,
      open_pnl,
      positions, // Always an array, never undefined
    };

    sendJson(res, PortfolioSummaryResponse, summaryData);

  } catch (e: any) {
    console.error('Portfolio summary error:', e);
    res.status(500).json({
      error: "PortfolioSummaryFailed",
      message: e.message || "Failed to fetch portfolio summary"
    });
  }
});

export default r;
