import React from 'react';
import { cn } from '@/utils/cn';

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  options: { value: string; label: string }[];
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, error, options, ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="text-sm font-medium text-foreground">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={cn(
            "w-full rounded-md border border-input bg-background px-3 py-2",
            "text-sm text-foreground placeholder:text-muted-foreground",
            "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
            "disabled:cursor-not-allowed disabled:opacity-50",
            error && "border-destructive focus:ring-destructive",
            className
          )}
          {...props}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
      </div>
    );
  }
);

Select.displayName = 'Select';

interface SelectItemProps {
  value: string;
  children: React.ReactNode;
}

export const SelectItem: React.FC<SelectItemProps> = ({ value, children }) => {
  return <option value={value}>{children}</option>;
};

interface SelectGroupProps {
  children: React.ReactNode;
}

export const SelectContent: React.FC<SelectGroupProps> = ({ children }) => {
  return <>{children}</>;
};

export const SelectGroup: React.FC<SelectGroupProps> = ({ children }) => {
  return <>{children}</>;
};

export const SelectValue: React.FC<{ placeholder?: string }> = () => {
  // This is handled by the native select element
  return null;
};

export const SelectTrigger: React.FC<{ children: React.ReactNode; className?: string }> = () => {
  // This is handled by the native select element
  return null;
};
