import { api } from './api';

export interface EvoSession {
  id: string;
  startedAt?: string;
  date?: string;
  status: string;
  bestFitness?: number;
  generations?: number;
  elapsed?: string;
}

/**
 * Fetches EvoTester history with safe error handling
 * for 404s and other common failures
 */
export async function fetchEvoHistory(): Promise<EvoSession[]> {
  try {
    const res = await api.get('/evotester/history');
    
    // Check if response is an array
    if (Array.isArray(res.data)) {
      return res.data;
    }
    
    // Check for nested data
    if (res.data?.items && Array.isArray(res.data.items)) {
      return res.data.items;
    }
    
    if (res.data?.data && Array.isArray(res.data.data)) {
      return res.data.data;
    }
    
    console.warn('Unexpected EvoTester history format:', res.data);
    return [];
  } catch (err: any) {
    // Handle 404 gracefully
    if (err?.response?.status === 404) {
      console.log('EvoTester history endpoint not available, returning empty array');
      return []; 
    }
    
    // Log other errors
    console.error('Failed to fetch EvoTester history:', err);
    throw err;
  }
}

/**
 * Fetches a specific EvoTester session by ID
 */
export async function fetchEvoSession(id: string): Promise<EvoSession | null> {
  try {
    const res = await api.get(`/evotester/${id}`);
    return res.data;
  } catch (err: any) {
    // Handle 404 gracefully
    if (err?.response?.status === 404) {
      return null;
    }
    
    console.error(`Failed to fetch EvoTester session ${id}:`, err);
    throw err;
  }
}

/**
 * Starts a new EvoTester session
 */
export async function startEvoSession(config: any): Promise<{ sessionId: string } | null> {
  try {
    const res = await api.post('/evotester/start', config);
    return { 
      sessionId: res.data?.session_id || res.data?.id || '' 
    };
  } catch (err) {
    console.error('Failed to start EvoTester session:', err);
    throw err;
  }
}

/**
 * Stops an active EvoTester session
 */
export async function stopEvoSession(id: string): Promise<boolean> {
  try {
    await api.post(`/evotester/${id}/stop`);
    return true;
  } catch (err) {
    console.error(`Failed to stop EvoTester session ${id}:`, err);
    return false;
  }
}
