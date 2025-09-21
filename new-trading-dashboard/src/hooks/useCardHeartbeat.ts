import { useEffect } from "react";

/**
 * Hook to track when dashboard cards are mounted
 * Helps identify which UI components are actually being used
 */
export function useCardHeartbeat(cardId: string) {
  useEffect(() => {
    // Send heartbeat when component mounts
    fetch("/api/telemetry/card-mounted", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cardId })
    }).catch((error) => {
      // Silently fail - telemetry shouldn't break the UI
      console.debug("Card heartbeat failed:", error);
    });
  }, [cardId]);
}

/**
 * Hook to get telemetry data about card usage
 */
export function useCardTelemetry() {
  return {
    async getCardStats() {
      try {
        const response = await fetch("/api/telemetry/cards");
        if (!response.ok) throw new Error("Failed to fetch telemetry");
        return await response.json();
      } catch (error) {
        console.warn("Failed to fetch card telemetry:", error);
        return { cards: [], total_unique_cards: 0, asOf: null };
      }
    },

    async resetStats() {
      try {
        const response = await fetch("/api/telemetry/reset", { method: "POST" });
        return response.ok;
      } catch (error) {
        console.warn("Failed to reset telemetry:", error);
        return false;
      }
    }
  };
}
