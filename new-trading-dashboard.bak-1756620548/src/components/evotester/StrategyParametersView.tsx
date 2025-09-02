import React, { useState } from 'react';
import { ArrowDown, ArrowUp, ChevronDown, ChevronUp } from 'lucide-react';
import { EvoStrategy } from '@/types/api.types';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';

interface StrategyParametersViewProps {
  strategy: EvoStrategy;
}

const StrategyParametersView: React.FC<StrategyParametersViewProps> = ({ strategy }) => {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    if (expandedSection === section) {
      setExpandedSection(null);
    } else {
      setExpandedSection(section);
    }
  };

  // Group parameters by category if they follow a pattern like "ma_period", "ma_type"
  const groupParameters = () => {
    const groups: Record<string, Record<string, any>> = {};
    const standalone: Record<string, any> = {};

    if (!strategy.parameters) {
      return { groups, standalone };
    }

    // Try to identify parameter groups
    Object.entries(strategy.parameters).forEach(([key, value]) => {
      const parts = key.split('_');
      if (parts.length > 1) {
        const groupName = parts[0];
        if (!groups[groupName]) {
          groups[groupName] = {};
        }
        const paramName = parts.slice(1).join('_');
        groups[groupName][paramName] = value;
      } else {
        standalone[key] = value;
      }
    });

    return { groups, standalone };
  };

  const { groups, standalone } = groupParameters();

  const renderValue = (value: any) => {
    if (typeof value === 'number') {
      return value.toString().includes('.') ? 
        value.toFixed(value > 100 ? 0 : value > 10 ? 1 : 4) : 
        value;
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    if (Array.isArray(value)) {
      return value.join(', ');
    }
    return value?.toString() || 'N/A';
  };

  const renderParameters = (params: Record<string, any>) => {
    return Object.entries(params).map(([key, value]) => (
      <div key={key} className="flex justify-between py-1 border-b border-gray-100 last:border-0">
        <span className="text-sm text-gray-600 capitalize">
          {key.replace(/_/g, ' ')}:
        </span>
        <span className="text-sm font-medium">{renderValue(value)}</span>
      </div>
    ));
  };

  return (
    <Tabs defaultValue="parameters">
      <TabsList className="w-full mb-4">
        <TabsTrigger value="parameters" className="flex-1">Parameters</TabsTrigger>
        <TabsTrigger value="performance" className="flex-1">Performance</TabsTrigger>
        <TabsTrigger value="json" className="flex-1">JSON</TabsTrigger>
      </TabsList>
      
      <TabsContent value="parameters">
        <div className="space-y-4">
          {/* Standalone parameters */}
          {Object.keys(standalone).length > 0 && (
            <div className="bg-gray-50 rounded-md p-3">
              <div 
                className="flex justify-between items-center cursor-pointer"
                onClick={() => toggleSection('general')}
              >
                <h3 className="text-md font-medium">General Parameters</h3>
                {expandedSection === 'general' ? 
                  <ChevronUp className="h-4 w-4" /> : 
                  <ChevronDown className="h-4 w-4" />
                }
              </div>
              
              {(expandedSection === 'general' || Object.keys(standalone).length < 5) && (
                <div className="mt-2 space-y-1">
                  {renderParameters(standalone)}
                </div>
              )}
            </div>
          )}
          
          {/* Grouped parameters */}
          {Object.entries(groups).map(([groupName, params]) => (
            <div key={groupName} className="bg-gray-50 rounded-md p-3">
              <div 
                className="flex justify-between items-center cursor-pointer"
                onClick={() => toggleSection(groupName)}
              >
                <h3 className="text-md font-medium capitalize">{groupName.replace(/_/g, ' ')} Parameters</h3>
                {expandedSection === groupName ? 
                  <ChevronUp className="h-4 w-4" /> : 
                  <ChevronDown className="h-4 w-4" />
                }
              </div>
              
              {(expandedSection === groupName || Object.keys(params).length < 5) && (
                <div className="mt-2 space-y-1">
                  {renderParameters(params)}
                </div>
              )}
            </div>
          ))}
        </div>
      </TabsContent>
      
      <TabsContent value="performance">
        <div className="bg-gray-50 rounded-md p-4">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Fitness Score:</span>
              <span className="font-medium text-blue-600">{strategy.fitness.toFixed(4)}</span>
            </div>
            
            {strategy.performance && (
              <>
                {strategy.performance.sharpeRatio !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Sharpe Ratio:</span>
                    <span className="font-medium">{strategy.performance.sharpeRatio.toFixed(2)}</span>
                  </div>
                )}
                
                {strategy.performance.sortinoRatio !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Sortino Ratio:</span>
                    <span className="font-medium">{strategy.performance.sortinoRatio.toFixed(2)}</span>
                  </div>
                )}
                
                {strategy.performance.winRate !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Win Rate:</span>
                    <span className="font-medium">{(strategy.performance.winRate * 100).toFixed(1)}%</span>
                  </div>
                )}
                
                {strategy.performance.profitFactor !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Profit Factor:</span>
                    <span className="font-medium">{strategy.performance.profitFactor.toFixed(2)}</span>
                  </div>
                )}
                
                {strategy.performance.maxDrawdown !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Max Drawdown:</span>
                    <span className="font-medium">{(strategy.performance.maxDrawdown * 100).toFixed(1)}%</span>
                  </div>
                )}
              </>
            )}
            
            {strategy.created && (
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Created:</span>
                <span className="font-medium">{new Date(strategy.created).toLocaleString()}</span>
              </div>
            )}
            
            {strategy.tags && strategy.tags.length > 0 && (
              <div className="pt-2">
                <span className="text-gray-600">Tags:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {strategy.tags.map((tag, index) => (
                    <span 
                      key={index}
                      className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded-full"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </TabsContent>
      
      <TabsContent value="json">
        <div className="bg-gray-50 rounded-md p-4">
          <pre className="text-xs overflow-auto max-h-[400px]">
            {JSON.stringify(strategy, null, 2)}
          </pre>
        </div>
      </TabsContent>
    </Tabs>
  );
};

export default StrategyParametersView;
