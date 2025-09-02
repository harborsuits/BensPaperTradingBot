import React from 'react';
import { EnhancedErrorBoundary } from './components/util/EnhancedErrorBoundary';
import DashboardPage from './pages/DashboardPage';

export default function App() {
  return (
    <EnhancedErrorBoundary>
      <DashboardPage />
    </EnhancedErrorBoundary>
  );
}

