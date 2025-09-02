import { ErrorBoundary } from 'react-error-boundary';
import React from 'react';

interface FallbackProps {
  error: Error;
  resetErrorBoundary: (...args: any[]) => void;
}

function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  return (
    <div className="p-4 rounded-lg border">
      <h2 className="font-semibold mb-2">Something went wrong</h2>
      <pre className="text-sm opacity-70 overflow-auto max-h-40">{error.message}</pre>
      <button 
        className="mt-3 px-3 py-1 rounded bg-black text-white" 
        onClick={resetErrorBoundary}
      >
        Retry
      </button>
    </div>
  );
}

interface EnhancedErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<FallbackProps>;
  onReset?: () => void;
}

export function EnhancedErrorBoundary({
  children,
  fallback = ErrorFallback,
  onReset,
}: EnhancedErrorBoundaryProps) {
  return (
    <ErrorBoundary 
      FallbackComponent={fallback} 
      onReset={onReset}
    >
      {children}
    </ErrorBoundary>
  );
}

export default EnhancedErrorBoundary;
