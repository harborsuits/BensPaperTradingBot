import axios from 'axios';
import { Alert, AlertFilterOptions, AlertsState } from '@/types/alerts.types';

// Helper function to make API requests, copied from api.ts to avoid circular dependencies
async function apiRequest<T>(config: any) {
  try {
    const api = axios.create({
      baseURL: '/api',
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    const response = await api(config);
    return { success: true, data: response.data };
  } catch (error: any) {
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message || 'Unknown error' 
    };
  }
};

/**
 * API service for working with alerts and notifications
 */
export const alertsApi = {
  /**
   * Get all alerts based on filter options
   */
  getAlerts: async (filters?: AlertFilterOptions) => {
    let queryParams = '';
    
    if (filters) {
      const params = new URLSearchParams();
      
      if (filters.severity && filters.severity.length > 0) {
        params.append('severity', filters.severity.join(','));
      }
      
      if (filters.source && filters.source.length > 0) {
        params.append('source', filters.source.join(','));
      }
      
      if (filters.status && filters.status.length > 0) {
        params.append('status', filters.status.join(','));
      }
      
      if (filters.timeRange) {
        params.append('from', filters.timeRange.from.toISOString());
        params.append('to', filters.timeRange.to.toISOString());
      }
      
      if (filters.search) {
        params.append('search', filters.search);
      }
      
      queryParams = `?${params.toString()}`;
    }
    
    return apiRequest<Alert[]>({
      url: `/alerts${queryParams}`,
      method: 'GET'
    });
  },
  
  /**
   * Get the current alerts state including unread count
   */
  getAlertsState: async () => {
    return apiRequest<AlertsState>({
      url: '/alerts/state',
      method: 'GET'
    });
  },
  
  /**
   * Get a specific alert by ID
   */
  getAlertById: async (id: string) => {
    return apiRequest<Alert>({
      url: `/alerts/${id}`,
      method: 'GET'
    });
  },
  
  /**
   * Update the status of an alert
   */
  updateAlertStatus: async (id: string, status: Alert['status']) => {
    return apiRequest<Alert>({
      url: `/alerts/${id}/status`,
      method: 'PATCH',
      data: { status }
    });
  },
  
  /**
   * Mark alerts as read/acknowledged
   */
  acknowledgeAlerts: async (ids: string[]) => {
    return apiRequest<{ success: boolean, count: number }>({
      url: '/alerts/acknowledge',
      method: 'POST',
      data: { ids }
    });
  },

  /** Acknowledge a single alert (matches live-api POST /api/alerts/:id/acknowledge) */
  acknowledge: async (id: string) => {
    return apiRequest<{ status: string }>({
      url: `/alerts/${id}/acknowledge`,
      method: 'POST'
    });
  },
  
  /**
   * Get recent notifications
   */
  getNotifications: async (limit = 10) => {
    // Use alerts endpoint as the notification source
    return apiRequest<Alert[]>({
      url: `/alerts?limit=${limit}`,
      method: 'GET'
    });
  },

  /** Explicit recent alerts helper */
  getRecent: async (limit = 10) => {
    return apiRequest<Alert[]>({
      url: `/alerts?limit=${limit}`,
      method: 'GET'
    });
  }
};
