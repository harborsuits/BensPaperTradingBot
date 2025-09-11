/**
 * Simple test card to verify Phase 3 EVO integration
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useEvoLedger, useEvoPoolStatus } from '@/hooks/useEvoQueries';

interface EvoTestCardProps {
  className?: string;
}

const EvoTestCard: React.FC<EvoTestCardProps> = ({ className = '' }) => {
  const { data: ledger, isLoading: ledgerLoading } = useEvoLedger();
  const { data: poolStatus, isLoading: poolLoading } = useEvoPoolStatus();

  return (
    <Card className={`${className} border-green-200`}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <span>🧬</span>
          <span>EVO Phase 3 Test</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium text-gray-600">Pool Status</h4>
            {poolLoading ? (
              <div className="text-sm text-gray-500">Loading...</div>
            ) : poolStatus ? (
              <div className="space-y-1">
                <div className="text-sm">
                  <span className="font-medium">Cap:</span> {(poolStatus.capPct * 100).toFixed(1)}%
                </div>
                <div className="text-sm">
                  <span className="font-medium">Used:</span> {(poolStatus.utilizationPct * 100).toFixed(1)}%
                </div>
                <div className="text-sm">
                  <span className="font-medium">Risk:</span>
                  <Badge className={`ml-1 text-xs ${
                    poolStatus.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                    poolStatus.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {poolStatus.riskLevel}
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="text-sm text-red-500">Failed to load</div>
            )}
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-600">Allocations</h4>
            {ledgerLoading ? (
              <div className="text-sm text-gray-500">Loading...</div>
            ) : ledger ? (
              <div className="space-y-1">
                <div className="text-sm">
                  <span className="font-medium">Active:</span> {ledger.rows?.filter(r => r.status === 'active').length || 0}
                </div>
                <div className="text-sm">
                  <span className="font-medium">Staged:</span> {ledger.rows?.filter(r => r.status === 'staged').length || 0}
                </div>
                <div className="text-sm">
                  <span className="font-medium">Total:</span> {ledger.rows?.length || 0}
                </div>
              </div>
            ) : (
              <div className="text-sm text-red-500">Failed to load</div>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2 pt-2 border-t">
          <Badge variant="outline" className="text-xs">
            PHASE 3 ACTIVE
          </Badge>
          <Badge variant="outline" className="text-xs">
            PAPER ONLY
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
};

export default EvoTestCard;
