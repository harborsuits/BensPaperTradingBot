import React from 'react';

export interface ErrorMessageProps {
  title?: string;
  message?: string;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  title = 'Something went wrong',
  message = 'Please try again later.',
}) => {
  return (
    <div className="rounded-md border border-red-400/30 bg-red-500/5 p-3 text-red-500">
      <div className="text-sm font-medium">{title}</div>
      {message && <div className="mt-1 text-xs text-red-400">{message}</div>}
    </div>
  );
};

export default ErrorMessage;


