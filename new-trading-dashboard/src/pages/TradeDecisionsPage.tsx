import React from 'react';
import { useDecisionsRecent } from '@/hooks/useDecisionsRecent';
import { Card } from '@/components/ui/Card';
import { TrendingUp } from 'lucide-react';

const TradeDecisionsPage: React.FC = () => {
  const { data: decisions } = useDecisionsRecent(100);

  return (
    <div className="w-full min-h-screen bg-background">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-foreground">Trade Decisions</h1>
            <div className="text-sm px-3 py-1 rounded-full bg-green-500/20 text-green-400">
              Live via WebSocket
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Real-time trading decisions and execution details
          </p>
        </div>

        {/* Main Content */}
        <div className="dashboard-container">
          {(!decisions || decisions.length === 0) ? (
            <div className="dashboard-section">
              <Card className="card">
                <div className="card-content">
                  <div className="text-center py-12">
                    <TrendingUp className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium mb-2">No decisions yet</h3>
                    <p className="text-muted-foreground">Trade decisions will appear here as they are made.</p>
                  </div>
                </div>
              </Card>
            </div>
          ) : (
            <div className="space-y-6">
              {decisions.map((d: any) => (
                <div key={String(d.trace_id || d.id)} className="dashboard-section">
                  <Card className="card">
                    <div className="card-header">
                      <div className="flex items-center justify-between">
                        <h3 className="card-title">{d.symbol} • {String(d.action || d.side).toUpperCase()}</h3>
                        <div className="text-sm text-muted-foreground">{d.createdAt || d.decidedAt || d.timestamp}</div>
                      </div>
                    </div>
                    <div className="card-content">
                      <p className="text-muted-foreground mb-4 leading-relaxed">{d.one_liner || d.reason || 'No reason provided'}</p>

                      {/* Tags and Status */}
                      <div className="flex flex-wrap gap-2 mb-4">
                        {(Array.isArray(d.reasons) ? d.reasons : []).slice(0,4).map((r: string) => (
                          <span key={r} className="px-3 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30">{r}</span>
                        ))}
                        {(Array.isArray(d.sources) ? d.sources : []).slice(0,4).map((s: string) => (
                          <span key={s} className="px-3 py-1 rounded-full text-xs bg-purple-500/20 text-purple-400 border border-purple-500/30">#{s}</span>
                        ))}
                        {d?.gates?.passed === false && (
                          <span className="px-3 py-1 rounded-full text-xs bg-red-500/20 text-red-400 border border-red-500/30">blocked</span>
                        )}
                        {d?.gates?.passed === true && (
                          <span className="px-3 py-1 rounded-full text-xs bg-green-500/20 text-green-400 border border-green-500/30">approved</span>
                        )}
                      </div>

                      {/* Plan Details */}
                      {d?.plan && (
                        <div className="p-4 bg-card/30 rounded-lg border border-border/50">
                          <h4 className="text-sm font-medium text-foreground mb-2">Order Plan</h4>
                          <p className="text-sm text-muted-foreground">
                            {String(d.plan.orderType || d.plan.type || '—')} •
                            Qty: {Number(d.plan.qty || d.size || 0)} •
                            {d.plan.limit ? `Limit: $${d.plan.limit}` : ''}
                            {d.plan.stop ? `Stop: $${d.plan.stop}` : ''}
                          </p>
                        </div>
                      )}
                    </div>
                  </Card>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TradeDecisionsPage;