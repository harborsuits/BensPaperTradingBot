import SafetyControls from '@/components/trading/SafetyControls';
import EnhancedPortfolioSummary from '@/components/portfolio/EnhancedPortfolioSummary';
import EnhancedTradeHistory from '@/components/portfolio/EnhancedTradeHistory';

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-slate-100">
      <div className="p-3 border-b bg-slate-50">
        <SafetyControls />
      </div>
      <div className="p-4 grid gap-4 md:grid-cols-2">
        <EnhancedPortfolioSummary />
        <EnhancedTradeHistory />
      </div>
    </div>
  );
}


