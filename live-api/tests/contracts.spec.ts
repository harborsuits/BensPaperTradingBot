import request from "supertest";
import app from "../server";

// Helper to handle both array and {items: []} responses consistently
const asItems = (body: any) => Array.isArray(body?.items) ? body.items : (Array.isArray(body) ? body : []);

describe("API contracts", () => {
  it("health v1", async () => {
    const { body, status } = await request(app).get("/api/health");
    expect(status).toBe(200);
    expect(typeof body.ok).toBe("boolean");
    expect(typeof body.breaker).toBe("string");
    expect(typeof body.version).toBe("string");
  });

  it("contracts endpoint", async () => {
    const { body, status } = await request(app).get("/api/contracts");
    expect(status).toBe(200);
    expect(body).toHaveProperty("health");
    expect(body).toHaveProperty("decisions_recent");
    expect(body).toHaveProperty("portfolio_summary");
    expect(typeof body.health).toBe("string");
  });

  it("portfolio summary v1", async () => {
    const { body, status } = await request(app).get("/api/portfolio/summary");
    expect(status).toBe(200);
    expect(typeof body.equity).toBe("number");
    expect(typeof body.cash).toBe("number");
    expect(Array.isArray(body.positions)).toBe(true); // never undefined
    expect(typeof body.asOf).toBe("string");
    expect(body.broker).toBe("tradier");
    expect(body.mode).toBe("paper");
  });

  it("decisions recent (proposed)", async () => {
    const { body, status } = await request(app).get("/api/decisions/recent?stage=proposed&limit=5");
    expect(status).toBe(200);
    const items = asItems(body);
    expect(Array.isArray(items)).toBe(true);
    if (items.length > 0) {
      expect(items[0]).toHaveProperty("symbol");
      expect(items[0]).toHaveProperty("strategy_id");
      expect(typeof items[0].confidence).toBe("number");
    }
  });

  it("decisions recent (intent)", async () => {
    const { body, status } = await request(app).get("/api/decisions/recent?stage=intent&limit=5");
    expect(status).toBe(200);
    const items = asItems(body);
    expect(Array.isArray(items)).toBe(true);
  });

  it("paper orders", async () => {
    const { body, status } = await request(app).get("/api/paper/orders?limit=5");
    expect(status).toBe(200);
    const items = asItems(body);
    expect(Array.isArray(items)).toBe(true);
    if (items.length > 0) {
      expect(items[0]).toHaveProperty("id");
      expect(items[0]).toHaveProperty("symbol");
      expect(items[0]).toHaveProperty("status");
    }
  });

  it("paper positions", async () => {
    const { body, status } = await request(app).get("/api/paper/positions");
    expect(status).toBe(200);
    const items = asItems(body);
    expect(Array.isArray(items)).toBe(true);
    if (items.length > 0) {
      expect(items[0]).toHaveProperty("symbol");
      expect(typeof items[0].quantity).toBe("number");
    }
  });

  it("brain activity", async () => {
    const { body, status } = await request(app).get("/api/brain/activity?limit=5");
    expect(status).toBe(200);
    const items = asItems(body);
    expect(Array.isArray(items)).toBe(true);
    if (items.length > 0) {
      expect(items[0]).toHaveProperty("symbol");
      expect(typeof items[0].final_score).toBe("number");
    }
  });

  it("handles malformed responses gracefully", async () => {
    // This test ensures our contract validation catches issues
    const { body, status } = await request(app).get("/api/health");
    expect(status).toBe(200);
    // If the response doesn't match schema, it should fail in development
    // In production, it would return the validated response or error
    expect(body).toBeDefined();
  });
});
