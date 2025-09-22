import React from 'react';
import { ArrowRight, CheckCircle, XCircle, Clock } from 'lucide-react';

interface StageData {
  name: string;
  count: number;
  status: 'success' | 'warning' | 'error' | 'processing';
  details?: string;
}

interface PipelineFlowDiagramProps {
  stages: StageData[];
  className?: string;
}

export const PipelineFlowDiagram: React.FC<PipelineFlowDiagramProps> = ({ stages, className = '' }) => {
  const getStatusIcon = (status: StageData['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-blue-500 animate-pulse" />;
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: StageData['status']) => {
    switch (status) {
      case 'success':
        return 'border-green-500 bg-green-500/10';
      case 'error':
        return 'border-red-500 bg-red-500/10';
      case 'processing':
        return 'border-blue-500 bg-blue-500/10';
      default:
        return 'border-yellow-500 bg-yellow-500/10';
    }
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between overflow-x-auto pb-4">
        {stages.map((stage, index) => (
          <React.Fragment key={stage.name}>
            <div className="flex flex-col items-center min-w-[120px]">
              <div className={`p-4 rounded-lg border-2 ${getStatusColor(stage.status)} transition-all`}>
                <div className="flex items-center gap-2 mb-2">
                  {getStatusIcon(stage.status)}
                  <span className="font-semibold text-sm">{stage.name}</span>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{stage.count}</div>
                  {stage.details && (
                    <div className="text-xs text-muted-foreground mt-1">{stage.details}</div>
                  )}
                </div>
              </div>
            </div>
            {index < stages.length - 1 && (
              <ArrowRight className="w-6 h-6 text-muted-foreground mx-2 flex-shrink-0" />
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default PipelineFlowDiagram;
