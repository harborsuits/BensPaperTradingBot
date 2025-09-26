import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// import { pricesStream, decisionsStream } from './lib/streams' // Commented out - file doesn't exist
import { Provider } from 'react-redux'
import { store } from './redux/store'
import { AuthProvider } from './context/AuthContext'
import { WebSocketProvider } from './contexts/WebSocketContext'
import { DataSyncProvider } from './contexts/DataSyncContext'
import { DATA_REFRESH_CONFIG } from './config/dataRefreshConfig'

// Initialize WebSocket connections
// pricesStream.start(); // Commented out - pricesStream not imported
// decisionsStream.start(); // Commented out - decisionsStream not imported

// MSW (Mock Service Worker) setup - conditionally enable
if (import.meta.env.VITE_USE_MSW === 'true') {
  console.log('[MSW] Enabled. Using mock backend.');
  import('./mocks/browser').then(({ worker }) => {
    worker.start();
  });
} else {
  console.log('[MSW] Disabled (VITE_USE_MSW!=true). Using real backend.');
}

// Create a client with optimized refresh settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: true, // Refresh data when tab becomes active
      refetchOnReconnect: 'always', // Refresh on network reconnect
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors (except 429)
        if (error?.status >= 400 && error?.status < 500 && error?.status !== 429) {
          return false;
        }
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: DATA_REFRESH_CONFIG.default?.staleTime || 20000,
      gcTime: 10 * 60 * 1000, // 10 minutes garbage collection
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <WebSocketProvider>
            <DataSyncProvider>
              <App />
            </DataSyncProvider>
          </WebSocketProvider>
        </AuthProvider>
      </QueryClientProvider>
    </Provider>
  </React.StrictMode>,
)