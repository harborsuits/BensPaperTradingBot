import React from 'react';

interface ErrorMessageProps {
  title?: string;
  message?: string;
  retry?: () => void;
}

export function ErrorMessage({ 
  title = 'Something went wrong', 
  message = 'There was an error loading the data. Please try again later.',
  retry
}: ErrorMessageProps) {
  return (
    <div className="rounded-md bg-destructive/10 p-4 my-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <svg className="h-5 w-5 text-destructive" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-destructive">{title}</h3>
          <div className="mt-2 text-sm text-destructive/80">
            <p>{message}</p>
          </div>
          {retry && (
            <div className="mt-4">
              <button
                type="button"
                onClick={retry}
                className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-destructive bg-destructive/10 hover:bg-destructive/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-destructive"
              >
                Try again
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 