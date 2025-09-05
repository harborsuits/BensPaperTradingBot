import React from 'react';
import { EnhancedErrorBoundary } from './components/util/EnhancedErrorBoundary';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage';
import SymbolPage from './pages/SymbolPage';

export default function App() {
  return (
    <EnhancedErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/symbol/:symbol" element={<SymbolPage />} />
        </Routes>
      </BrowserRouter>
    </EnhancedErrorBoundary>
  );
}

