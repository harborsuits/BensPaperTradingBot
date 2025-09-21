const express = require('express');
const r = express.Router();

// In-memory store for card mount counts
// In production, you might want to persist this to a database
const cardMountCounts = new Map();

r.post("/telemetry/card-mounted", (req, res) => {
  try {
    const { cardId } = req.body;
    if (!cardId || typeof cardId !== 'string') {
      return res.status(400).json({ error: "Invalid cardId" });
    }

    const currentCount = cardMountCounts.get(cardId) || 0;
    cardMountCounts.set(cardId, currentCount + 1);

    res.json({ ok: true, count: currentCount + 1 });
  } catch (e) {
    res.status(500).json({ error: "TelemetryError", message: e.message });
  }
});

r.get("/telemetry/cards", (_req, res) => {
  console.log('Telemetry endpoint called');
  try {
    const sortedCards = Array.from(cardMountCounts.entries())
      .sort((a, b) => b[1] - a[1]) // Sort by count descending
      .map(([cardId, count]) => ({ cardId, count }));

    res.json({
      cards: sortedCards,
      total_unique_cards: cardMountCounts.size,
      asOf: new Date().toISOString()
    });
  } catch (e) {
    console.log('Telemetry error:', e.message);
    res.status(500).json({ error: "TelemetryFetchError", message: e.message });
  }
});

// Reset endpoint for testing
r.post("/telemetry/reset", (_req, res) => {
  cardMountCounts.clear();
  res.json({ ok: true, message: "Telemetry reset" });
});

module.exports = r;
