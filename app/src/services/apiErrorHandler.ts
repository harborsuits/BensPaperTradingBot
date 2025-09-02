import { AxiosError } from 'axios';

// Track 404s to avoid spamming console
const logged404s: Record<string, boolean> = {};

// Helper to identify context endpoints
const isContextEndpoint = (url?: string): boolean =>
  !!url && (
    url.includes('/context/regime') ||
    url.includes('/context/features') ||
    url.includes('/context/sentiment') ||
    url.includes('/context/metrics')
  );

// Create appropriate fallbacks based on endpoint
const fallbackFor = (url?: string): any => {
  if (!url) return {};
  
  if (url.includes('/context/regime')) return { regime: 'unknown', confidence: 0 };
  if (url.includes('/context/features')) return { features: [] };
  if (url.includes('/context/sentiment')) return { score: null, history: [], anomalies: [] };
  if (url.includes('/strategies')) return { items: [] };
  if (url.includes('/trades')) return { items: [] };
  if (url.includes('/portfolio')) return { 
    summary: { total_equity: 0, cash_balance: 0, daily_pl: 0, daily_pl_percent: 0 },
    positions: []
  };
  
  return {};
};

// Generate a unique key for each endpoint to avoid repeated logging
const getRequestKey = (config: any): string => {
  return `${config?.method || 'GET'}-${config?.url || ''}`;
};

// Handle errors globally
export const handleApiError = (error: unknown): Promise<any> => {
  const err = error as AxiosError;
  const status = err?.response?.status;
  const url = err?.config?.url;
  
  // Handle 401 Unauthorized
  if (status === 401) {
    localStorage.removeItem('auth_token');
    window.location.href = '/login';
    return Promise.reject(error);
  }
  
  // Handle 404s with fallbacks for known endpoints
  if (status === 404) {
    // Log once per endpoint
    const requestKey = getRequestKey(err.config);
    if (!logged404s[requestKey]) {
      console.warn(`API 404: ${url || '[unknown path]'}`);
      logged404s[requestKey] = true;
    }
    
    // Return graceful fallbacks for common endpoints
    if (isContextEndpoint(url) || url?.includes('/strategies') || url?.includes('/trades')) {
      return Promise.resolve({ 
        data: fallbackFor(url), 
        status: 200, 
        headers: {}, 
        config: err.config 
      });
    }
  }
  
  return Promise.reject(error);
};

export const resetErrorLog = (): void => {
  Object.keys(logged404s).forEach(key => {
    delete logged404s[key];
  });
};
