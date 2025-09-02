import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import OpportunityScanner from './components/OpportunityScanner';
import './styles/responsive.css';
import './index.css';

const queryClient = new QueryClient();

const rootEl = document.getElementById('root');
if (rootEl) {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <OpportunityScanner />
      </QueryClientProvider>
    </React.StrictMode>
  );
}
