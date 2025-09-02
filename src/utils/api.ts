import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { mockAPI } from './mockAPI';

// Define types for Vite's import.meta.env
declare global {
  interface ImportMetaEnv {
    DEV: boolean;
    VITE_API_BASE_URL?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

// Use mock API in development, real API in production
const isDevelopment = import.meta.env.DEV || !import.meta.env.VITE_API_BASE_URL;

// Export the appropriate API client
export const api = isDevelopment ? mockAPI : axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

// Only set up these interceptors if using the real API
if (!isDevelopment && 'interceptors' in api) {
  const axiosInstance = api as AxiosInstance;
  
  // Request interceptor for adding headers, auth tokens, etc.
  axiosInstance.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // You can add authentication headers here
      // Example: config.headers.Authorization = `Bearer ${localStorage.getItem('token')}`;
      return config;
    },
    (error: AxiosError) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor for handling errors globally
  axiosInstance.interceptors.response.use(
    (response: AxiosResponse) => {
      return response;
    },
    (error: AxiosError) => {
      // Handle specific error statuses, e.g. redirect to login on 401
      if (error.response?.status === 401) {
        // Handle unauthorized access
        console.error('Unauthorized access');
      }
      return Promise.reject(error);
    }
  );
} 