import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { ArrowUpIcon, ArrowDownIcon, TrendingUpIcon, TrendingDownIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useOptionsPositions } from '@/hooks/useOptionsData';

export default function OptionsTable() {
  const { data: positions, isLoading } = useOptionsPositions();

  const getDaysToExpiry = (expiration: string) => {
    const exp = new Date(expiration);
    const today = new Date();
    const diff = exp.getTime() - today.getTime();
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
  };

  const formatOptionSymbol = (symbol: string, underlying: string, strike: number, type: string) => {
    return `${underlying} ${strike} ${type.toUpperCase()}`;
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Options Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            <div className="h-10 bg-gray-200 rounded"></div>
            <div className="h-10 bg-gray-200 rounded"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Options Positions</span>
          <Badge variant="outline" className="ml-2">
            {positions?.length || 0} Active
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {positions && positions.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Option</TableHead>
                <TableHead className="text-center">Type</TableHead>
                <TableHead className="text-center">Qty</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Current</TableHead>
                <TableHead className="text-center">DTE</TableHead>
                <TableHead className="text-right">P&L</TableHead>
                <TableHead className="text-center">Strategy</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((position) => {
                const dte = getDaysToExpiry(position.expiration);
                const isShort = position.quantity < 0;
                
                return (
                  <TableRow key={position.id}>
                    <TableCell className="font-medium">
                      {formatOptionSymbol(
                        position.symbol,
                        position.underlying,
                        position.strike,
                        position.optionType
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge 
                        variant={position.optionType === 'call' ? 'default' : 'secondary'}
                        className="w-12"
                      >
                        {position.optionType === 'call' ? 'C' : 'P'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center font-mono">
                      {position.quantity}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      ${position.entryPrice.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      ${position.currentPrice.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge 
                        variant={dte <= 7 ? 'destructive' : dte <= 21 ? 'secondary' : 'outline'}
                      >
                        {dte}d
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className={cn(
                        "font-medium",
                        position.pnl >= 0 ? "text-green-600" : "text-red-600"
                      )}>
                        ${Math.abs(position.pnl).toFixed(2)}
                      </div>
                      <div className={cn(
                        "text-xs flex items-center justify-end",
                        position.pnlPercent >= 0 ? "text-green-600" : "text-red-600"
                      )}>
                        {position.pnlPercent >= 0 ? (
                          <TrendingUpIcon className="w-3 h-3 mr-1" />
                        ) : (
                          <TrendingDownIcon className="w-3 h-3 mr-1" />
                        )}
                        {Math.abs(position.pnlPercent).toFixed(1)}%
                      </div>
                    </TableCell>
                    <TableCell className="text-center">
                      {position.metadata?.strategy && (
                        <Badge variant="outline" className="text-xs">
                          {position.metadata.strategy.replace('_', ' ')}
                        </Badge>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No options positions yet</p>
            <p className="text-sm mt-2">Options strategies will appear here when executed</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
