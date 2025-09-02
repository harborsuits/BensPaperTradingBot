import React from 'react';
import SafetyTest from '@/components/tests/SafetyTest';
import { EnhancedErrorBoundary } from '@/components/util/EnhancedErrorBoundary';

const TestPage: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Testing Utilities</h1>
      
      <EnhancedErrorBoundary>
        <SafetyTest />
      </EnhancedErrorBoundary>
      
      <div className="mt-8 p-4 border rounded-md">
        <h2 className="text-xl font-bold mb-4">Debug Tools</h2>
        <div className="space-y-2">
          <button 
            onClick={() => {
              const status = window._wsDebug && window._wsDebug();
              console.log('WebSocket status:', status);
            }}
            className="px-3 py-2 bg-blue-600 text-white rounded"
          >
            Check WebSocket Status
          </button>
          
          <button 
            onClick={() => {
              console.log('Test 404 handling...');
              fetch('/api/context/nonexistent').catch(e => console.log('Fetch error handled'));
            }}
            className="px-3 py-2 bg-amber-600 text-white rounded ml-2"
          >
            Test 404 Handling
          </button>
          
          <button
            onClick={() => {
              // Trigger multiple WebSocket connections/HMR
              console.log('Testing WebSocket cleanup...');
              console.log('Simulate HMR by checking current state');
              window._wsDebug && window._wsDebug();
              console.log('WebSocket should be properly cleaned up on HMR');
            }}
            className="px-3 py-2 bg-green-600 text-white rounded ml-2"
          >
            Test WebSocket HMR
          </button>
        </div>
        
        <pre className="mt-4 p-3 bg-gray-100 rounded text-xs">
          {`
// In dev tools console:

// 1. WebSocket diagnostics
window._wsDebug();

// 2. Try injecting a message
window.__injectWsMessage({
  type: 'trade_executed',
  data: { action: 'BUY', quantity: 5, symbol: 'AAPL', price: 198.42 },
  timestamp: new Date().toISOString()
});

// 3. Test toast notifications
window.showTestToast?.();
          `}
        </pre>
      </div>
    </div>
  );
};

// Add to dev tools
if (typeof window !== 'undefined') {
  (window as any).showTestToast = () => {
    const { toast } = require('react-hot-toast');
    toast.success('This is a test toast!');
  };
}

export default TestPage;
