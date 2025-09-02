import React from 'react';
import { AlertTriangle, Search, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { useHealth } from '@/hooks/useHealth';

type EmptyStateProps = {
  title: string;
  description?: string;
  icon?: 'error' | 'empty' | 'loading';
  action?: React.ReactNode;
  className?: string;
  showHealth?: boolean;
};

const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  icon = 'empty',
  action,
  className = '',
  showHealth = false,
}) => {
  const { data: health } = useHealth();
  const isHealthy = health?.breaker === 'GREEN';

  const Icon = 
    icon === 'error' ? AlertTriangle :
    icon === 'loading' ? Loader2 : 
    Search;

  return (
    <div className={`p-4 text-center space-y-2 ${className}`}>
      <Icon className="mx-auto h-8 w-8 mb-2 text-muted-foreground opacity-50" />
      <div className="font-medium">{title}</div>
      {description && <div className="text-sm text-muted-foreground">{description}</div>}
      {showHealth && (
        <div className="flex justify-center mt-2">
          <Badge variant={isHealthy ? "default" : "destructive"} className="text-xs">
            Health: {health?.breaker || 'UNKNOWN'}
          </Badge>
        </div>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
};

export default EmptyState;
