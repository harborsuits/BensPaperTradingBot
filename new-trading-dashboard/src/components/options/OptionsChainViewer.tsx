import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Input } from '@/components/ui/Input';
import { ArrowRight, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Strike {
  strike: number;
  callBid: number;
  callAsk: number;
  callVolume: number;
  callOpenInterest: number;
  callIV: number;
  callDelta: number;
  putBid: number;
  putAsk: number;
  putVolume: number;
  putOpenInterest: number;
  putIV: number;
  putDelta: number;
}

interface OptionsChain {
  symbol: string;
  underlyingPrice: number;
  expirations: string[];
  strikes: Strike[];
}

export default function OptionsChainViewer() {
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [selectedExpiration, setSelectedExpiration] = useState<string>('');
  
  const { data: chain, isLoading } = useQuery<OptionsChain>({
    queryKey: ['options', 'chain', selectedSymbol, selectedExpiration],
    queryFn: async () => {
      // Mock data for demonstration
      const expirations = [
        '2024-12-20',
        '2024-12-27',
        '2025-01-17',
        '2025-02-21'
      ];
      
      return {
        symbol: selectedSymbol,
        underlyingPrice: 450.25,
        expirations,
        strikes: [
          {
            strike: 440,
            callBid: 12.50, callAsk: 12.70, callVolume: 1500, callOpenInterest: 5000, callIV: 0.18, callDelta: 0.75,
            putBid: 2.10, putAsk: 2.30, putVolume: 800, putOpenInterest: 3000, putIV: 0.19, putDelta: -0.25
          },
          {
            strike: 445,
            callBid: 8.20, callAsk: 8.40, callVolume: 2200, callOpenInterest: 8000, callIV: 0.17, callDelta: 0.65,
            putBid: 3.10, putAsk: 3.30, putVolume: 1200, putOpenInterest: 4500, putIV: 0.18, putDelta: -0.35
          },
          {
            strike: 450,
            callBid: 4.50, callAsk: 4.70, callVolume: 5500, callOpenInterest: 15000, callIV: 0.16, callDelta: 0.50,
            putBid: 4.40, putAsk: 4.60, putVolume: 5000, putOpenInterest: 14000, putIV: 0.16, putDelta: -0.50
          },
          {
            strike: 455,
            callBid: 2.30, callAsk: 2.50, callVolume: 3200, callOpenInterest: 10000, callIV: 0.17, callDelta: 0.35,
            putBid: 6.80, putAsk: 7.00, putVolume: 2500, putOpenInterest: 8000, putIV: 0.17, putDelta: -0.65
          },
          {
            strike: 460,
            callBid: 1.10, callAsk: 1.30, callVolume: 1800, callOpenInterest: 6000, callIV: 0.18, callDelta: 0.25,
            putBid: 10.50, putAsk: 10.70, putVolume: 1000, putOpenInterest: 4000, putIV: 0.18, putDelta: -0.75
          }
        ]
      };
    },
    enabled: !!selectedExpiration
  });

  // Set initial expiration when data loads
  React.useEffect(() => {
    if (chain?.expirations && !selectedExpiration) {
      setSelectedExpiration(chain.expirations[0]);
    }
  }, [chain?.expirations, selectedExpiration]);

  const getMoneyness = (strike: number, underlying: number) => {
    const diff = ((strike - underlying) / underlying) * 100;
    if (Math.abs(diff) < 1) return 'ATM';
    return diff > 0 ? 'OTM' : 'ITM';
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}k`;
    return volume.toString();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Options Chain</CardTitle>
        <div className="flex gap-4 mt-4">
          <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="SPY">SPY</SelectItem>
              <SelectItem value="QQQ">QQQ</SelectItem>
              <SelectItem value="AAPL">AAPL</SelectItem>
              <SelectItem value="TSLA">TSLA</SelectItem>
            </SelectContent>
          </Select>
          
          {chain?.expirations && (
            <Select value={selectedExpiration} onValueChange={setSelectedExpiration}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Select expiration" />
              </SelectTrigger>
              <SelectContent>
                {chain.expirations.map(exp => {
                  const date = new Date(exp);
                  const days = Math.ceil((date.getTime() - Date.now()) / (1000 * 60 * 60 * 24));
                  return (
                    <SelectItem key={exp} value={exp}>
                      {exp} ({days}d)
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        {isLoading ? (
          <div className="animate-pulse space-y-2">
            <div className="h-10 bg-gray-200 rounded"></div>
            <div className="h-10 bg-gray-200 rounded"></div>
          </div>
        ) : chain ? (
          <div className="space-y-4">
            <div className="text-center">
              <p className="text-sm text-gray-500">Underlying Price</p>
              <p className="text-2xl font-bold">${chain.underlyingPrice.toFixed(2)}</p>
            </div>
            
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead colSpan={6} className="text-center bg-green-50">
                      Calls
                    </TableHead>
                    <TableHead className="text-center bg-gray-50">Strike</TableHead>
                    <TableHead colSpan={6} className="text-center bg-red-50">
                      Puts
                    </TableHead>
                  </TableRow>
                  <TableRow>
                    {/* Calls */}
                    <TableHead className="text-right">OI</TableHead>
                    <TableHead className="text-right">Vol</TableHead>
                    <TableHead className="text-right">IV</TableHead>
                    <TableHead className="text-right">Bid</TableHead>
                    <TableHead className="text-right">Ask</TableHead>
                    <TableHead className="text-right">Δ</TableHead>
                    
                    {/* Strike */}
                    <TableHead className="text-center font-bold">Strike</TableHead>
                    
                    {/* Puts */}
                    <TableHead className="text-right">Δ</TableHead>
                    <TableHead className="text-right">Bid</TableHead>
                    <TableHead className="text-right">Ask</TableHead>
                    <TableHead className="text-right">IV</TableHead>
                    <TableHead className="text-right">Vol</TableHead>
                    <TableHead className="text-right">OI</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {chain.strikes.map((strike) => {
                    const moneyness = getMoneyness(strike.strike, chain.underlyingPrice);
                    const isATM = moneyness === 'ATM';
                    
                    return (
                      <TableRow key={strike.strike} className={cn(isATM && "bg-blue-50")}>
                        {/* Calls */}
                        <TableCell className="text-right text-xs">
                          {formatVolume(strike.callOpenInterest)}
                        </TableCell>
                        <TableCell className="text-right text-xs">
                          {formatVolume(strike.callVolume)}
                        </TableCell>
                        <TableCell className="text-right text-xs">
                          {(strike.callIV * 100).toFixed(0)}%
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          ${strike.callBid.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          ${strike.callAsk.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right text-xs">
                          {strike.callDelta.toFixed(2)}
                        </TableCell>
                        
                        {/* Strike */}
                        <TableCell className="text-center font-bold">
                          <div className="flex items-center justify-center gap-2">
                            ${strike.strike}
                            {isATM && <Badge variant="secondary" className="text-xs">ATM</Badge>}
                          </div>
                        </TableCell>
                        
                        {/* Puts */}
                        <TableCell className="text-right text-xs">
                          {strike.putDelta.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          ${strike.putBid.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          ${strike.putAsk.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right text-xs">
                          {(strike.putIV * 100).toFixed(0)}%
                        </TableCell>
                        <TableCell className="text-right text-xs">
                          {formatVolume(strike.putVolume)}
                        </TableCell>
                        <TableCell className="text-right text-xs">
                          {formatVolume(strike.putOpenInterest)}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
            
            <div className="text-xs text-gray-500 mt-4">
              <p>OI: Open Interest | Vol: Volume | IV: Implied Volatility | Δ: Delta</p>
              <p className="mt-1">Click on bid/ask prices to place orders</p>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            Select an expiration date to view options chain
          </div>
        )}
      </CardContent>
    </Card>
  );
}
