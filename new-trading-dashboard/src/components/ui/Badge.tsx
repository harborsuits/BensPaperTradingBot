import React from 'react';
import { cn } from '@/utils/cn';

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline' | 'success';
  size?: 'default' | 'sm' | 'lg';
}

const Badge: React.FC<BadgeProps> = ({
  className,
  variant = 'default',
  size = 'default',
  ...props
}) => {
  const variantClasses = {
    default: 'bg-primary text-primary-foreground',
    secondary: 'bg-secondary text-secondary-foreground',
    destructive: 'bg-destructive text-destructive-foreground',
    outline: 'border border-border bg-transparent',
    success: 'bg-green-500/10 text-green-500',
  };

  const sizeClasses = {
    default: 'px-2.5 py-0.5 text-xs',
    sm: 'px-2 py-0.5 text-xs',
    lg: 'px-3 py-1 text-sm',
  };

  return (
    <div
      className={cn(
        'inline-flex items-center rounded-full font-medium',
        sizeClasses[size],
        variantClasses[variant],
        className
      )}
      {...props}
    />
  );
};

export { Badge };
