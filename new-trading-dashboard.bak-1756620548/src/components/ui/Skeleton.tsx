import React from 'react';

interface SkeletonProps {
  height?: number;
  width?: number | string;
  radius?: number;
}

export function Skeleton({ height = 16, width = '100%', radius = 10 }: SkeletonProps) {
  return (
    <div
      aria-hidden
      style={{
        height,
        width,
        borderRadius: radius,
        background: 'linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.12), rgba(255,255,255,0.06))',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.2s ease-in-out infinite',
      }}
    />
  );
}

export default Skeleton;
