import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import { WebSocketProvider } from '@/contexts/WebSocketContext';

// Initialize MSW only when explicitly enabled
async function setupMocks() {
  const useMsw = (import.meta.env as any)['VITE_USE_MSW'] === 'true';
  if (import.meta.env.DEV && useMsw) {
    try {
      const { worker } = await import('./mocks/browser');
      await worker.start({
        onUnhandledRequest: 'bypass',
        quiet: true,
        serviceWorker: { 
          url: '/mockServiceWorker.js' // Ensure same origin/port
        }
      });
      console.log('[MSW] Mock service worker activated');
    } catch (error) {
      console.error('[MSW] Failed to initialize:', error);
    }
  } else if (import.meta.env.DEV) {
    console.log('[MSW] Disabled (VITE_USE_MSW!=true). Using real backend.');
  }
  return Promise.resolve();
}

// Top-level await ensures MSW starts before React
await setupMocks();
import MainLayout from '@/components/layouts/MainLayout';
import DashboardPage from '@/pages/DashboardPage';
import PortfolioPage from '@/pages/PortfolioPage';
import StrategiesPage from '@/pages/StrategiesPage';
import MarketContextPage from '@/pages/MarketContextPage';
import TradeDecisionsPage from '@/pages/TradeDecisionsPage';
import MarketDataPage from '@/pages/MarketDataPage';
import SafetyPage from '@/pages/SafetyPage';
import ScannerPage from '@/pages/ScannerPage';
import LogsPage from '@/pages/LogsPage';
import EvoTesterPage from '@/pages/EvoTesterPage';
import TestPage from '@/pages/TestPage';
import ApiTest from '@/pages/ApiTest';
import LoginPage from '@/pages/LoginPage';
import MarketDashboard from '@/pages/MarketDashboard';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from './lib/queryClient';
import RQDevtoolsToggle from './shared/RQDevtoolsToggle';
import './index.css';
import './styles/comfort.css';

// Create router with v7 flags enabled
const router = createBrowserRouter(
  [
    // Public
    { path: '/login', element: <LoginPage /> },
    
    // App Shell
    {
      path: '/',
      element: <MainLayout />,
      children: [
        { index: true, element: <DashboardPage /> },
        { path: 'portfolio', element: <PortfolioPage /> },
        { path: 'strategies', element: <StrategiesPage /> },
        { path: 'decisions', element: <TradeDecisionsPage /> },
        { path: 'context', element: <MarketContextPage /> },
        { path: 'market-data', element: <MarketDataPage /> },
        { path: 'scanner', element: <ScannerPage /> },
        { path: 'safety', element: <SafetyPage /> },
        { path: 'evotester', element: <EvoTesterPage /> },
        { path: 'logs', element: <LogsPage /> },
        { path: 'market', element: <MarketDashboard /> },
        { path: 'test', element: <TestPage /> },
        { path: 'api-test', element: <ApiTest /> }
      ]
    }
  ],
  {
    future: {
      v7_startTransition: true,
      v7_relativeSplatPath: true,
    },
  }
);

const rootEl = document.getElementById('root');
if (rootEl) {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <WebSocketProvider>
            <RouterProvider router={router} />
          </WebSocketProvider>
          {(import.meta.env as any).VITE_RQ_DEVTOOLS !== 'false' && <RQDevtoolsToggle />}
        </AuthProvider>
      </QueryClientProvider>
    </React.StrictMode>
  );
}
