import React, { useState } from 'react';
import { 
  ArrowUp, 
  ArrowDown, 
  Briefcase,
  PieChart,
  AlertTriangle,
  InfoIcon
} from 'lucide-react';
import { format } from 'date-fns';
import { Position } from '@/types/api.types';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip';
import DataTable from '@/components/ui/DataTable';

interface EnhancedPositionTableProps {
  positions: Position[];
  account: 'paper' | 'live';
  onPositionSelect: (position: Position) => void;
  onClosePosition?: (symbol: string) => void;
  lastUpdate?: Date | null;
}

const EnhancedPositionTable: React.FC<EnhancedPositionTableProps> = ({
  positions,
  account,
  onPositionSelect,
  onClosePosition,
  lastUpdate
}) => {
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  // Create position table columns with strategy information
  const positionColumns = [
    {
      header: 'Symbol',
      accessor: 'symbol',
      cell: (position: Position) => (
        <div className="flex items-center">
          <span className="font-medium">{position.symbol}</span>
          {position.is_leveraged && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <AlertTriangle size={14} className="ml-1 text-amber-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Leveraged position</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      ),
    },
    {
      header: 'Quantity',
      accessor: 'quantity',
      cell: (position: Position) => (
        <div className="flex items-center">
          <span
            className={position.quantity > 0 ? 'text-green-500' : 'text-red-500'}
          >
            {position.quantity > 0 ? '+ ' : '- '}
            {Math.abs(position.quantity).toLocaleString()}
          </span>
        </div>
      ),
    },
    {
      header: 'Avg. Cost',
      accessor: 'average_cost',
      cell: (position: Position) => (
        <span>${position.average_cost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
      ),
    },
    {
      header: 'Last Price',
      accessor: 'last_price',
      cell: (position: Position) => (
        <div className="flex items-center">
          <span className="mr-1">
            ${position.last_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span 
            className={
              position.price_change_pct >= 0 
                ? 'text-green-500 flex items-center text-xs' 
                : 'text-red-500 flex items-center text-xs'
            }
          >
            {position.price_change_pct >= 0 ? (
              <ArrowUp size={14} className="mr-0.5" />
            ) : (
              <ArrowDown size={14} className="mr-0.5" />
            )}
            {Math.abs(position.price_change_pct).toFixed(2)}%
          </span>
        </div>
      ),
    },
    {
      header: 'Strategy',
      accessor: 'strategy_name',
      cell: (position: Position) => (
        <div className="flex items-center">
          {position.strategy_name ? (
            <Badge variant="outline" className="bg-slate-100 dark:bg-slate-800">
              {position.strategy_name}
            </Badge>
          ) : (
            <span className="text-muted-foreground text-xs">Manual</span>
          )}
          {position.strategy_id && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <InfoIcon size={14} className="ml-1 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <div className="space-y-1">
                    <p className="font-medium">{position.strategy_name}</p>
                    {position.strategy_type && (
                      <p className="text-xs">Type: {position.strategy_type}</p>
                    )}
                    {position.entry_reason && (
                      <p className="text-xs">Reason: {position.entry_reason}</p>
                    )}
                    {position.entry_timestamp && (
                      <p className="text-xs">Entered: {format(new Date(position.entry_timestamp), 'MMM d, h:mm a')}</p>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      ),
    },
    {
      header: 'P/L',
      accessor: 'unrealized_pl',
      cell: (position: Position) => (
        <div className="flex flex-col">
          <span
            className={
              position.unrealized_pl >= 0 ? 'text-green-600' : 'text-red-600'
            }
          >
            {position.unrealized_pl >= 0 ? '+' : ''}$
            {position.unrealized_pl.toLocaleString(undefined, {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </span>
          <span
            className={
              position.unrealized_pl_pct >= 0
                ? 'text-green-600 text-xs'
                : 'text-red-600 text-xs'
            }
          >
            {position.unrealized_pl_pct >= 0 ? '+' : ''}
            {position.unrealized_pl_pct.toFixed(2)}%
          </span>
        </div>
      ),
    },
    {
      header: 'Allocation',
      accessor: 'allocation_pct',
      cell: (position: Position) => (
        <div className="flex items-center">
          <PieChart size={14} className="mr-1 text-muted-foreground" />
          <span>{position.allocation_pct.toFixed(2)}%</span>
        </div>
      ),
    },
    {
      header: 'Actions',
      accessor: '_actions',
      cell: (position: Position) => (
        <div className="flex items-center space-x-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onPositionSelect(position)}
          >
            <InfoIcon size={14} />
          </Button>
          {onClosePosition && (
            <Button
              size="sm"
              variant="destructive"
              className={`opacity-0 transition-opacity ${
                hoveredRow === position.symbol ? 'opacity-100' : ''
              }`}
              onClick={(e) => {
                e.stopPropagation();
                if (onClosePosition) onClosePosition(position.symbol);
              }}
            >
              Close
            </Button>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-3">
      {lastUpdate && (
        <div className="flex justify-end">
          <span className="text-xs text-muted-foreground">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </span>
        </div>
      )}
      
      {positions.length > 0 ? (
        <DataTable
          data={positions}
          columns={positionColumns}
          keyExtractor={(item) => `${item.symbol}-${item.account}`}
          onRowMouseEnter={(position) => setHoveredRow(position.symbol)}
          onRowMouseLeave={() => setHoveredRow(null)}
          onRowClick={(position) => onPositionSelect(position)}
          searchable
          searchPlaceholder="Search positions..."
        />
      ) : (
        <div className="py-12 flex flex-col items-center justify-center space-y-2">
          <Briefcase size={32} className="text-muted-foreground" />
          <p className="text-muted-foreground">No open positions in your {account} account</p>
        </div>
      )}
    </div>
  );
};

export default EnhancedPositionTable;
