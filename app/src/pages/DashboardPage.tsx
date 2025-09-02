import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { TrendingUp, ChevronRight, AlertTriangle } from 'lucide-react';

import { portfolioApi, decisionApi } from '@/services/api';
import { qk } from '@/services/qk';
import { toPortfolio } from '@/services/normalize';
import UniverseSwitcher from '@/components/UniverseSwitcher';

// Import new trading components
import LoopStripBanner from '@/components/trading/LoopStripBanner';
import CandidateCard from '@/components/trading/CandidateCard';
import EnhancedDecisionCard from '@/components/trading/EnhancedDecisionCard';
import OpenOrdersPanel from '@/components/trading/OpenOrdersPanel';
import ActivityTicker from '@/components/trading/ActivityTicker';
import MarketAwarenessLine from '@/components/trading/MarketAwarenessLine';
import AutoRunnerStrip from '@/components/trading/AutoRunnerStrip';
import CandidatesQuickList from '@/components/trading/CandidatesQuickList';
import { SimpleCard } from '@/components/ui/SimpleCard';

// Import narrative utilities
import { formatDecisionNarrative } from '@/utils/decisionNarrative';

const DashboardPage: React.FC = () => {
  const { data: portfolioData } = useQuery({
    queryKey: qk.portfolio('paper'),
    queryFn: () => portfolioApi.getPortfolio('paper'),
    refetchInterval: 15_000,
  });
  const paperPortfolio = toPortfolio(portfolioData?.data);

  const { data: recentDecisions } = useQuery({
    queryKey: qk.decisions(4),
    queryFn: () => decisionApi.getLatestDecisions(4),
    refetchInterval: 7_000,
  });

  return (
    <div className="container py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <UniverseSwitcher />
      </div>

      <div className="space-y-6">
        {/* Portfolio Summary Card */}
        <SimpleCard title="Portfolio Summary" action={<Link to="/portfolio" className="text-sm text-primary flex items-center">View details <ChevronRight size={16} /></Link>}>
          <div className="flex flex-wrap gap-4">
            <div className="border border-border rounded-md p-3 flex-1 min-w-[300px]">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Paper Trading</span>
              </div>
              {paperPortfolio ? (
                <div className="flex flex-wrap gap-4">
                  <div>
                    <div className="text-muted-foreground text-sm">Total Equity</div>
                    <div className="font-medium text-lg">${paperPortfolio.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-sm">Cash Balance</div>
                    <div className="font-medium text-lg">${paperPortfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Loading portfolio...</p>
              )}
            </div>
          </div>
        </SimpleCard>

        {/* Activity Loop Rail */}
        <div className="space-y-4">
          <AutoRunnerStrip />
          <CandidatesQuickList />
          <LoopStripBanner />
          <MarketAwarenessLine className="px-2" />
          <CandidateCard />

          <SimpleCard title="Recent Decisions" action={<Link to="/decisions" className="text-sm text-primary flex items-center">View all <ChevronRight size={16} /></Link>}>
            <div className="min-h-[120px]">
              {recentDecisions?.data && recentDecisions.data.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {recentDecisions.data.map((decision) => (
                    <EnhancedDecisionCard
                      key={decision.id}
                      decision={formatDecisionNarrative(decision)}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  <TrendingUp className="mx-auto h-8 w-8 mb-2 opacity-50" />
                  <p>No recent trade decisions</p>
                </div>
              )}
            </div>
          </SimpleCard>

          <SimpleCard title="Open Orders">
            <OpenOrdersPanel />
          </SimpleCard>

          <ActivityTicker />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
