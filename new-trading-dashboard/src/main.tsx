import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { pricesStream, decisionsStream } from './lib/streams'
import { Provider } from 'react-redux'
import { store } from './redux/store'
import { AuthProvider } from './context/AuthContext'
import { WebSocketProvider } from './contexts/WebSocketContext'

// Initialize WebSocket connections
pricesStream.start();
decisionsStream.start();

// MSW (Mock Service Worker) setup - conditionally enable
if (import.meta.env.VITE_USE_MSW === 'true') {
  console.log('[MSW] Enabled. Using mock backend.');
  import('./mocks/browser').then(({ worker }) => {
    worker.start();
  });
} else {
  console.log('[MSW] Disabled (VITE_USE_MSW!=true). Using real backend.');
}

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <WebSocketProvider>
            <App />
          </WebSocketProvider>
        </AuthProvider>
      </QueryClientProvider>
    </Provider>
  </React.StrictMode>,
)