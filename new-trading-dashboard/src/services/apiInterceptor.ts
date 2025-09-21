import axios, { AxiosError, AxiosRequestConfig } from 'axios';

// Store for seen 404s to avoid duplicates
const seen404s = new Set<string>();

/**
 * Generate a stable key for a request to use for deduplication
 */
export const keyFromRequest = (config: AxiosRequestConfig): string => {
  return [
    config?.method?.toUpperCase() || 'GET', 
    config?.baseURL || '', 
    config?.url || '', 
    JSON.stringify(config?.params || {}),
    JSON.stringify(config?.data || {})
  ].join(' ');
};

/**
 * Create fallback data for common endpoints
 */
export const fallbackForEndpoint = (url?: string): unknown => {
  if (!url) return {};
  
  if (url.includes('/context/regime')) return { regime: 'unknown', confidence: 0 };
  if (url.includes('/context/features')) return { features: [] };
  if (url.includes('/context/sentiment')) return { score: null, history: [], anomalies: [] };
  if (url.includes('/strategies')) return { items: [] };
  if (url.includes('/evotester/history')) return [];
  
  return {};
};

/**
 * Handle API errors gracefully
 */
export const handleApiError = (error: unknown): Promise<any> => {
  const err = error as AxiosError;
  const status = err?.response?.status;
  
  // Handle 401 Unauthorized
  if (status === 401) {
    localStorage.removeItem('auth_token');
    window.location.href = '/login';
    return Promise.reject(error);
  }
  
  // Let other errors pass through
  return Promise.reject(error);
};

/**
 * Clear the 404 log (useful for testing)
 */
export const clear404Log = (): void => {
  seen404s.clear();
};

/**
 * Apply interceptors to an axios instance
 */
export const applyApiInterceptors = (axiosInstance: typeof axios): void => {
  // Add request interceptor for authentication
  axiosInstance.interceptors.request.use((config) => {
    const token = localStorage.getItem('auth_token');
    
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  });
  
  // Add response interceptor for error handling
  axiosInstance.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => handleApiError(error)
  );
};
