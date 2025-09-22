import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import { showSuccessToast, showErrorToast } from '@/utils/toast';
import { useWebSocketMessage, useWebSocketChannel } from '@/services/websocket';

interface BotData {
  rank: number;
  id: string;
  strategy: string;
  symbol: string;
  generation: number;
  returnPct: number;
  returnDollar: number;
  currentCapital: number;
  trades: number;
  winRate: number;
}

interface CompetitionStatus {
  id: string;
  status: string;
  stats: {
    activeBots: number;
    totalReturn: number;
    totalTrades: number;
    daysLeft: number;
    hoursToNextReallocation: number;
  };
  leaderboard: BotData[];
}

interface AIBotCompetitionProps {
  className?: string;
}

const AIBotCompetition: React.FC<AIBotCompetitionProps> = ({ className = '' }) => {
  console.log('[AIBotCompetition] Component rendering');
  const queryClient = useQueryClient();
  const [activeCompetitionId, setActiveCompetitionId] = useState<string | null>(null);
  
  // Subscribe to bot competition channel
  useWebSocketChannel('bot-competition');
  
  // Handle WebSocket messages
  useWebSocketMessage((message) => {
    if (message.type === 'bot-competition-update' && message.data) {
      // Update the query cache with real-time data
      queryClient.setQueryData(['bot-competition', message.data.id], message.data);
      
      // Update active competition if needed
      if (!activeCompetitionId && message.data.status === 'active') {
        setActiveCompetitionId(message.data.id);
      }
    }
  });

  // Fetch active competitions
  const { data: competitionsData, error: competitionsError } = useQuery({
    queryKey: ['bot-competitions', 'active'],
    queryFn: async () => {
      console.log('[AIBotCompetition] Fetching active competitions...');
      const res = await fetch('/api/bot-competition/active');
      if (!res.ok) throw new Error('Failed to fetch competitions');
      const data = await res.json();
      console.log('[AIBotCompetition] Active competitions response:', data);
      return data;
    },
    refetchInterval: 5000 // Refresh every 5 seconds
  });
  
  if (competitionsError) {
    console.error('[AIBotCompetition] Error fetching competitions:', competitionsError);
  }

  // Set active competition ID
  useEffect(() => {
    if (competitionsData?.competitions?.length > 0 && !activeCompetitionId) {
      setActiveCompetitionId(competitionsData.competitions[0].id);
    }
  }, [competitionsData, activeCompetitionId]);

  // Fetch competition status
  const { data: competitionStatus, isLoading } = useQuery({
    queryKey: ['bot-competition', activeCompetitionId],
    queryFn: async () => {
      const res = await fetch(`/api/bot-competition/${activeCompetitionId}/status`);
      if (!res.ok) throw new Error('Failed to fetch competition status');
      return res.json() as Promise<CompetitionStatus>;
    },
    enabled: !!activeCompetitionId,
    refetchInterval: 3000 // Refresh every 3 seconds for real-time updates
  });

  // Start competition mutation
  const startCompetition = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/bot-competition/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          durationDays: 7,
          initialCapitalMin: 50,    // Everyone gets exactly $50
          initialCapitalMax: 50,    // Fair and uniform starting point
          totalPoolCapital: 5000,   // 100 bots * $50 = $5000
          winnerBonus: 0.2,
          loserPenalty: 0.5,
          reallocationIntervalHours: 1,
          botCount: 100            // Create 100 bots!
        })
      });
      if (!res.ok) throw new Error('Failed to start competition');
      return res.json();
    },
    onSuccess: (data) => {
      showSuccessToast('Competition started successfully!');
      setActiveCompetitionId(data.competition.id);
      queryClient.invalidateQueries({ queryKey: ['bot-competitions'] });
    },
    onError: () => {
      showErrorToast('Failed to start competition');
    }
  });

  // Reallocate capital mutation
  const reallocateCapital = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bot-competition/${activeCompetitionId}/reallocate`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to reallocate capital');
      return res.json();
    },
    onSuccess: () => {
      showSuccessToast('Capital reallocated successfully!');
      queryClient.invalidateQueries({ queryKey: ['bot-competition', activeCompetitionId] });
    },
    onError: () => {
      showErrorToast('Failed to reallocate capital');
    }
  });

  // End competition mutation
  const endCompetition = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bot-competition/${activeCompetitionId}/end`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to end competition');
      return res.json();
    },
    onSuccess: () => {
      showSuccessToast('Competition ended!');
      setActiveCompetitionId(null);
      queryClient.invalidateQueries({ queryKey: ['bot-competitions'] });
    },
    onError: () => {
      showErrorToast('Failed to end competition');
    }
  });

  const isCompetitionActive = competitionStatus?.status === 'active';
  const stats = competitionStatus?.stats || {
    activeBots: 0,
    totalReturn: 0,
    totalTrades: 0,
    daysLeft: 0,
    hoursToNextReallocation: 0
  };

  return (
    <div className={`p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">🤖 AI Bot Competition</h3>
          <p className="text-sm text-gray-600">Evolved strategies compete with micro-capital allocations</p>
        </div>
        <div className="flex space-x-2">
          {!isCompetitionActive ? (
            <button
              onClick={() => startCompetition.mutate()}
              disabled={startCompetition.isPending}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50 flex items-center"
            >
              {startCompetition.isPending ? (
                <Loader2 className="animate-spin mr-2 h-4 w-4" />
              ) : (
                '▶️ '
              )}
              Start Competition
            </button>
          ) : (
            <>
              <button
                onClick={() => reallocateCapital.mutate()}
                disabled={reallocateCapital.isPending}
                className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 disabled:opacity-50 flex items-center"
              >
                {reallocateCapital.isPending && <Loader2 className="animate-spin mr-2 h-4 w-4" />}
                🔄 Reallocate
              </button>
              <button
                onClick={() => endCompetition.mutate()}
                disabled={endCompetition.isPending}
                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 disabled:opacity-50 flex items-center"
              >
                {endCompetition.isPending && <Loader2 className="animate-spin mr-2 h-4 w-4" />}
                ⏹️ End Round
              </button>
            </>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <Loader2 className="animate-spin h-8 w-8" />
        </div>
      ) : (
        <>
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-lg font-bold text-blue-600">{stats.activeBots}</div>
              <div className="text-xs text-gray-600">Active Bots</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="text-lg font-bold text-green-600">
                {stats.totalReturn >= 0 ? '+' : ''}{stats.totalReturn.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600">Total Return</div>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <div className="text-lg font-bold text-purple-600">{stats.totalTrades}</div>
              <div className="text-xs text-gray-600">Total Trades</div>
            </div>
            <div className="text-center p-3 bg-orange-50 rounded-lg">
              <div className="text-lg font-bold text-orange-600">{Math.ceil(stats.daysLeft)}d</div>
              <div className="text-xs text-gray-600">Days Left</div>
            </div>
            <div className="text-center p-3 bg-indigo-50 rounded-lg">
              <div className="text-lg font-bold text-indigo-600">
                {Math.floor(stats.hoursToNextReallocation * 60)}m
              </div>
              <div className="text-xs text-gray-600">Next Reallocation</div>
            </div>
          </div>

          {/* Leaderboard */}
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-medium mb-3">
              🏆 Leaderboard 
              {competitionStatus?.leaderboard && competitionStatus.leaderboard.length > 10 && 
                <span className="text-sm font-normal text-gray-600 ml-2">
                  (Top 10 of {competitionStatus.leaderboard.length} bots)
                </span>
              }
            </h4>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {competitionStatus?.leaderboard?.slice(0, 10).map((bot) => (
                <div
                  key={bot.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-lg font-bold text-gray-400">#{bot.rank}</span>
                    <div>
                      <div className="font-medium">{bot.strategy}</div>
                      <div className="text-xs text-gray-500">
                        {bot.symbol} • Gen {bot.generation} • {bot.trades} trades
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${bot.returnPct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {bot.returnPct >= 0 ? '+' : ''}{bot.returnPct.toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">
                      ${bot.currentCapital.toFixed(2)}
                    </div>
                    {bot.winRate > 0 && (
                      <div className="text-xs text-gray-500">
                        Win: {bot.winRate.toFixed(0)}%
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {(!competitionStatus?.leaderboard || competitionStatus.leaderboard.length === 0) && (
                <div className="text-center text-gray-500 py-8">
                  No bots competing yet. Start a competition to begin!
                </div>
              )}
            </div>
          </div>

          {/* Competition Info */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium mb-2">🎯 Competition Rules</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• 100 bots compete with exactly $50 each</li>
              <li>• Total pool: $5,000 (100 × $50)</li>
              <li>• Winners get +20% more capital (snowball effect)</li>
              <li>• Losers get -50% less capital</li>
              <li>• Capital reallocates every hour automatically</li>
              <li>• Competition runs for 7 days</li>
              <li>• Triggered by significant news sentiment (AI curiosity)</li>
              <li>• Only the best strategies survive and grow!</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default AIBotCompetition;