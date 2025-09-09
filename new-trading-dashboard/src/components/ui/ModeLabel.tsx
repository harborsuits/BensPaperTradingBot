import React from 'react';
import { Badge } from '@/components/ui/Badge';
import { FlaskConical, Target, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

export type TradingMode = 'research' | 'competition' | 'validation';

interface ModeLabelProps {
  mode: TradingMode;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  status?: 'active' | 'completed' | 'failed' | 'pending';
  className?: string;
}

const ModeLabel: React.FC<ModeLabelProps> = ({
  mode,
  size = 'md',
  showIcon = true,
  status,
  className = ''
}) => {
  const getModeConfig = (mode: TradingMode) => {
    switch (mode) {
      case 'research':
        return {
          label: 'RESEARCH',
          color: 'bg-purple-100 text-purple-800 border-purple-300',
          icon: FlaskConical,
          description: 'EvoTester experiments with segregated capital'
        };
      case 'competition':
        return {
          label: 'COMPETITION',
          color: 'bg-blue-100 text-blue-800 border-blue-300',
          icon: Target,
          description: 'Main trading competition with full capital'
        };
      case 'validation':
        return {
          label: 'VALIDATION',
          color: 'bg-green-100 text-green-800 border-green-300',
          icon: CheckCircle,
          description: 'Testing promoted strategies before full deployment'
        };
      default:
        return {
          label: 'UNKNOWN',
          color: 'bg-gray-100 text-gray-800 border-gray-300',
          icon: AlertTriangle,
          description: 'Unknown trading mode'
        };
    }
  };

  const getStatusConfig = (status?: string) => {
    switch (status) {
      case 'active':
        return {
          icon: CheckCircle,
          color: 'text-green-600'
        };
      case 'completed':
        return {
          icon: CheckCircle,
          color: 'text-green-600'
        };
      case 'failed':
        return {
          icon: XCircle,
          color: 'text-red-600'
        };
      case 'pending':
        return {
          icon: AlertTriangle,
          color: 'text-yellow-600'
        };
      default:
        return null;
    }
  };

  const modeConfig = getModeConfig(mode);
  const statusConfig = getStatusConfig(status);
  const IconComponent = modeConfig.icon;
  const StatusIconComponent = statusConfig?.icon;

  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2'
  };

  const iconSizeClasses = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  };

  return (
    <div className={`inline-flex items-center space-x-2 ${className}`}>
      <Badge
        variant="outline"
        className={`${modeConfig.color} ${sizeClasses[size]} font-medium`}
      >
        {showIcon && (
          <IconComponent className={`${iconSizeClasses[size]} mr-1`} />
        )}
        {modeConfig.label}
        {status && StatusIconComponent && (
          <StatusIconComponent className={`${iconSizeClasses[size]} ml-1 ${statusConfig?.color}`} />
        )}
      </Badge>
    </div>
  );
};

export default ModeLabel;
