import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { EnhancedErrorBoundary } from './components/util/EnhancedErrorBoundary';
import MainLayout from '@/components/layouts/MainLayout';
import DashboardPage from '@/pages/DashboardPage';
import PortfolioPage from '@/pages/PortfolioPage';
import StrategiesPage from '@/pages/StrategiesPage';
import TradeDecisionsPage from '@/pages/TradeDecisions';
import MarketDataPage from '@/pages/MarketDataPage';
import LogsPage from '@/pages/LogsPage';
import EvoTesterPage from '@/pages/EvoTesterPage';

export default function App() {
  return (
    <EnhancedErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<MainLayout />}> 
            <Route index element={<DashboardPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/strategies" element={<StrategiesPage />} />
            <Route path="/decisions" element={<TradeDecisionsPage />} />
            <Route path="/market" element={<MarketDataPage />} />
            <Route path="/logs" element={<LogsPage />} />
            <Route path="/evotester" element={<EvoTesterPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </EnhancedErrorBoundary>
  );
}

