import React from 'react';

export const PositionSkeleton: React.FC = () => {
  return (
    <div className="w-full space-y-3">
      {Array.from({ length: 5 }).map((_, index) => (
        <div key={index} className="flex items-center gap-4 p-3 animate-pulse">
          <div className="w-16 h-6 bg-muted rounded-md"></div>
          <div className="w-20 h-6 bg-muted rounded-md"></div>
          <div className="w-24 h-6 bg-muted rounded-md"></div>
          <div className="w-20 h-6 bg-muted rounded-md"></div>
          <div className="w-20 h-6 bg-muted rounded-md"></div>
          <div className="w-16 h-6 bg-muted rounded-md"></div>
          <div className="ml-auto w-24 h-8 bg-muted rounded-md"></div>
        </div>
      ))}
    </div>
  );
};

export const SummarySkeleton: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 animate-pulse">
      <div className="bg-card border border-border rounded-lg p-4 space-y-3">
        <div className="w-24 h-5 bg-muted rounded-md"></div>
        <div className="w-36 h-8 bg-muted rounded-md"></div>
        <div className="w-20 h-5 bg-muted rounded-md"></div>
      </div>
      <div className="bg-card border border-border rounded-lg p-4 space-y-3">
        <div className="w-24 h-5 bg-muted rounded-md"></div>
        <div className="w-36 h-8 bg-muted rounded-md"></div>
        <div className="w-20 h-5 bg-muted rounded-md"></div>
      </div>
      <div className="bg-card border border-border rounded-lg p-4 space-y-3">
        <div className="w-24 h-5 bg-muted rounded-md"></div>
        <div className="w-36 h-8 bg-muted rounded-md"></div>
        <div className="w-20 h-5 bg-muted rounded-md"></div>
      </div>
    </div>
  );
};

export const ChartSkeleton: React.FC = () => {
  return (
    <div className="h-[300px] w-full animate-pulse bg-card border border-border rounded-lg p-4">
      <div className="w-32 h-6 bg-muted rounded-md mb-4"></div>
      <div className="h-[250px] w-full bg-muted rounded-md"></div>
    </div>
  );
};
