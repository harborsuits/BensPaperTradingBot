import React from 'react';
import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom';
import { EnhancedErrorBoundary } from './components/util/EnhancedErrorBoundary';
import MainLayout from '@/components/layouts/MainLayout';
import DashboardPage from '@/pages/DashboardPage';
import PortfolioPage from '@/pages/PortfolioPage';
import TradeDecisionsPage from '@/pages/TradeDecisions';
import MarketDataPage from '@/pages/MarketDataPage';
import LogsPage from '@/pages/LogsPage';
import EvoTesterPage from '@/pages/EvoTesterPage';
import NewsPage from '@/pages/NewsPage';
import { Outlet } from 'react-router-dom';

export default function App() {
  const router = createBrowserRouter([
    {
      element: <MainLayout />,
      children: [
        { index: true, element: <DashboardPage /> },
        { path: '/portfolio', element: <PortfolioPage /> },
        { path: '/decisions', element: <TradeDecisionsPage /> },
        { path: '/market', element: <MarketDataPage /> },
        { path: '/logs', element: <LogsPage /> },
        { path: '/evotester', element: <EvoTesterPage /> },
        { path: '/news', element: <NewsPage /> },
        { path: '/news/:symbol', element: <NewsPage /> },
      ],
    },
    { path: '*', element: <Navigate to="/" replace /> },
  ], {
    future: {
      v7_startTransition: true,
      v7_relativeSplatPath: true,
    },
  });

  return (
    <EnhancedErrorBoundary>
      <RouterProvider router={router} future={{ v7_startTransition: true }} />
    </EnhancedErrorBoundary>
  );
}

