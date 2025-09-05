import React from 'react';

type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg';

const sizeToClass: Record<SpinnerSize, string> = {
  xs: 'h-3 w-3 border-2',
  sm: 'h-4 w-4 border-2',
  md: 'h-6 w-6 border-2',
  lg: 'h-8 w-8 border-2',
};

export interface SpinnerProps {
  size?: SpinnerSize;
  className?: string;
}

export const Spinner: React.FC<SpinnerProps> = ({ size = 'md', className = '' }) => {
  const sizeClass = sizeToClass[size];
  return (
    <span
      className={`inline-block animate-spin rounded-full border-muted-foreground/30 border-t-primary ${sizeClass} ${className}`}
      aria-label="Loading"
      role="status"
    />
  );
};

export default Spinner;


