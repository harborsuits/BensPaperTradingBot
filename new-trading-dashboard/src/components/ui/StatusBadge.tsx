import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/utils/cn';

// Define badge styles and variants using class-variance-authority
const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
  {
    variants: {
      variant: {
        default: "bg-primary/10 text-primary",
        secondary: "bg-secondary/10 text-secondary",
        bull: "bg-bull/20 text-bull",
        bear: "bg-bear/20 text-bear",
        neutral: "bg-neutral/20 text-neutral",
        warning: "bg-warning/20 text-warning",
        info: "bg-info/20 text-info",
        highImpact: "bg-highImpact/20 text-highImpact",
      },
      size: {
        default: "text-xs px-2.5 py-0.5",
        sm: "text-[10px] px-2 py-0.5",
        lg: "text-sm px-3 py-1",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

// Define props for the StatusBadge component
export interface StatusBadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  pulse?: boolean;
  withDot?: boolean;
  dotColor?: string;
}

// Status badge component implementation
const StatusBadge = React.forwardRef<HTMLDivElement, StatusBadgeProps>(
  ({ className, variant, size, pulse = false, withDot = false, dotColor, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(badgeVariants({ variant, size }), className)}
        {...props}
      >
        {withDot && (
          <span 
            className={cn(
              "mr-1.5 inline-block h-2 w-2 rounded-full",
              dotColor || (
                variant === 'bull' ? "bg-bull" :
                variant === 'bear' ? "bg-bear" :
                variant === 'warning' ? "bg-warning" :
                variant === 'info' ? "bg-info" :
                variant === 'highImpact' ? "bg-highImpact" :
                "bg-primary"
              ),
              pulse && "animate-pulse"
            )}
          />
        )}
        {children}
      </div>
    );
  }
);

StatusBadge.displayName = "StatusBadge";

export { StatusBadge, badgeVariants };
