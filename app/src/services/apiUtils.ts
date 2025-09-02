import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import { ApiResponse } from '@/types/api.types';

// Export the apiRequest function that's used by all API services
export async function apiRequest<T>(config: AxiosRequestConfig): Promise<ApiResponse<T>> {
  // Get the API instance with token handling
  const api = getApiInstance();
  
  try {
    const response = await api(config);
    return { success: true, data: response.data };
  } catch (error) {
    const axiosError = error as AxiosError<{detail?: string}>;
    return { 
      success: false, 
      error: axiosError.response?.data?.detail || axiosError.message || 'Unknown error' 
    };
  }
}

// Create and configure an axios instance with auth handling
function getApiInstance() {
  const api = axios.create({
    baseURL: '/api', // This will be proxied by Vite to the backend
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Add request interceptor for authentication
  api.interceptors.request.use((config) => {
    const token = localStorage.getItem('auth_token');
    
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  });

  // Add response interceptor for error handling
  api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      // Handle 401 Unauthorized errors (token expired)
      if (error.response?.status === 401) {
        localStorage.removeItem('auth_token');
        window.location.href = '/login';
      }
      
      return Promise.reject(error);
    }
  );

  return api;
}
