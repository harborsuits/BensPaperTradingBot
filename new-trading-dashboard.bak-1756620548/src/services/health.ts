import { api } from './api';

/**
 * Simple HTTP health check endpoint that can be used 
 * as a fallback when WebSocket ping/pong fails
 */
export async function pingApi(): Promise<void> {
  try {
    await api.get('/health');
    // Request succeeded, API is reachable
    return;
  } catch (error) {
    console.warn('Health check failed:', error);
    throw error;
  }
}

/**
 * Check if the API is alive and track response time
 * Can be used for diagnostics
 */
export async function checkApiHealth(): Promise<{ ok: boolean, latency: number }> {
  const start = Date.now();
  
  try {
    await api.get('/ping');
    return {
      ok: true,
      latency: Date.now() - start
    };
  } catch (error) {
    return {
      ok: false,
      latency: Date.now() - start
    };
  }
}