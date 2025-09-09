/**
 * ============================================
 * [CARD: ACTIVE SESSIONS LIST]
 * Central hub for evolution test management - running experiments, session tracking, progress monitoring
 * ============================================
 */

import React, { useState } from 'react';
import { RefreshCw, Clock, BarChart3, Play, Pause, Square, TrendingUp, Activity, Target, Filter, Search, MoreHorizontal, Eye, Settings } from 'lucide-react';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { evoTesterApi } from '@/services/api';
import { showSuccessToast, showErrorToast } from '@/utils/toast.js';
import styles from './ActiveSessionsList.module.css';

interface Session {
  id: string;
  date: string;
  bestFitness: number;
  status?: string;
  progress?: number;
  currentGeneration?: number;
  totalGenerations?: number;
  symbols?: string[];
  strategyCount?: number;
  avgFitness?: number;
  bestStrategy?: string;
  marketRegime?: string;
  capital?: number;
  pnl?: number;
  riskLevel?: string;
  startTime?: string;
  estimatedCompletion?: string;
}

interface ActiveSessionsListProps {
  sessions: Session[];
  onSelectSession: (sessionId: string) => void;
  activeSessionId?: string;
  isLoading?: boolean;
}

const ActiveSessionsList: React.FC<ActiveSessionsListProps> = ({
  sessions,
  onSelectSession,
  activeSessionId,
  isLoading = false
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'date' | 'fitness' | 'progress'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showDetails, setShowDetails] = useState<string | null>(null);
  // Function to format the elapsed time
  const formatElapsedTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();

    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else {
      const hours = Math.floor(diffMins / 60);
      const mins = diffMins % 60;
      return `${hours}h ${mins}m ago`;
    }
  };

  // Filter and sort sessions
  const filteredAndSortedSessions = sessions
    .filter(session => {
      const matchesSearch = searchTerm === '' ||
        session.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        session.symbols?.some(symbol => symbol.toLowerCase().includes(searchTerm.toLowerCase()));

      const matchesStatus = statusFilter === 'all' || session.status === statusFilter;

      return matchesSearch && matchesStatus;
    })
    .sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'date':
          comparison = new Date(a.date).getTime() - new Date(b.date).getTime();
          break;
        case 'fitness':
          comparison = a.bestFitness - b.bestFitness;
          break;
        case 'progress':
          comparison = (a.progress || 0) - (b.progress || 0);
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

  // Handle pausing a session
  const handlePauseSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation(); // Prevent selecting the session when clicking the button
    
    try {
      const response = await evoTesterApi.pauseEvoTest(sessionId);
      if (response.success) {
        showSuccessToast('Session paused successfully');
      } else {
        showErrorToast('Failed to pause session');
      }
    } catch (err) {
      console.error('Error pausing session:', err);
      showErrorToast('Error pausing session');
    }
  };

  // Handle resuming a session
  const handleResumeSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation(); // Prevent selecting the session when clicking the button

    try {
      // Using the resume endpoint with the correct parameter names
      const response = await evoTesterApi.resumeEvoTest(sessionId);

      if (response.success) {
        showSuccessToast('Session resumed successfully');
      } else {
        showErrorToast('Failed to resume session');
      }
    } catch (err) {
      console.error('Error resuming session:', err);
      showErrorToast('Error resuming session');
    }
  };

  // Handle stopping a session
  const handleStopSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation(); // Prevent selecting the session when clicking the button

    try {
      const response = await evoTesterApi.stopEvoTest(sessionId);
      if (response.success) {
        showSuccessToast('Session stopped successfully');
      } else {
        showErrorToast('Failed to stop session');
      }
    } catch (err) {
      console.error('Error stopping session:', err);
      showErrorToast('Error stopping session');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-40">
        <RefreshCw className="h-5 w-5 animate-spin text-gray-400" />
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-40">
        <div className="text-gray-500 mb-2">No active sessions</div>
        <div className="text-xs text-gray-400">Start a new evolution to see real-time progress</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters and Search */}
      <div className="flex flex-col sm:flex-row gap-3 p-3 bg-gray-50 rounded-lg">
        <div className="flex-1 relative">
          <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search sessions or symbols..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div className="flex gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="paused">Paused</option>
            <option value="completed">Completed</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'date' | 'fitness' | 'progress')}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="date">Sort by Date</option>
            <option value="fitness">Sort by Fitness</option>
            <option value="progress">Sort by Progress</option>
          </select>

          <Button
            size="sm"
            variant="outline"
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="px-3"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </Button>
        </div>
      </div>

      {/* Session Statistics */}
      {sessions.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-blue-50 rounded-lg">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">{sessions.length}</div>
            <div className="text-xs text-gray-600">Total Sessions</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-green-600">
              {sessions.filter(s => s.status === 'running').length}
            </div>
            <div className="text-xs text-gray-600">Running</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-yellow-600">
              {sessions.filter(s => s.status === 'completed').length}
            </div>
            <div className="text-xs text-gray-600">Completed</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">
              {Math.max(...sessions.map(s => s.bestFitness || 0)).toFixed(3)}
            </div>
            <div className="text-xs text-gray-600">Best Fitness</div>
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="space-y-3">
        {filteredAndSortedSessions.map((session) => (
          <div
            key={session.id}
            className={`border rounded-lg transition-all duration-200 ${
              activeSessionId === session.id
                ? 'border-blue-500 bg-blue-50 shadow-md'
                : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
            }`}
          >
            {/* Main Session Header */}
            <div
              className="p-4 cursor-pointer"
              onClick={() => onSelectSession(session.id)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center flex-1">
                  <BarChart3 className="h-5 w-5 mr-3 text-blue-500 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold text-gray-900 truncate">
                      {session.id.substring(0, 12)}...
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Clock className="h-3 w-3 text-gray-400" />
                      <span className="text-xs text-gray-500">{formatElapsedTime(session.date)}</span>
                      {session.marketRegime && (
                        <>
                          <span className="text-gray-300">•</span>
                          <Badge variant="outline" className="text-xs">
                            {session.marketRegime}
                          </Badge>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 ml-4">
                  {session.status && (
                    <StatusBadge
                      variant={session.status === 'running' ? 'bull' :
                               session.status === 'paused' ? 'warning' :
                               session.status === 'completed' ? 'neutral' : 'bear'}
                    >
                      {session.status}
                    </StatusBadge>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowDetails(showDetails === session.id ? null : session.id);
                    }}
                    className="p-1"
                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                <div className="text-center">
                  <div className="text-sm font-semibold text-blue-600">
                    {session.bestFitness.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500">Best Fitness</div>
                </div>

                {session.avgFitness && (
                  <div className="text-center">
                    <div className="text-sm font-semibold text-green-600">
                      {session.avgFitness.toFixed(4)}
                    </div>
                    <div className="text-xs text-gray-500">Avg Fitness</div>
                  </div>
                )}

                {session.strategyCount && (
                  <div className="text-center">
                    <div className="text-sm font-semibold text-purple-600">
                      {session.strategyCount}
                    </div>
                    <div className="text-xs text-gray-500">Strategies</div>
                  </div>
                )}

                {session.symbols && (
                  <div className="text-center">
                    <div className="text-sm font-semibold text-orange-600">
                      {session.symbols.length}
                    </div>
                    <div className="text-xs text-gray-500">Symbols</div>
                  </div>
                )}
              </div>

              {/* Progress Bar */}
              {(session.progress !== undefined && session.currentGeneration !== undefined && session.totalGenerations !== undefined) && (
                <div className="mb-3">
                  <div className="flex justify-between items-center text-xs text-gray-500 mb-2">
                    <span>Generation {session.currentGeneration}/{session.totalGenerations}</span>
                    <span>{Math.round(session.progress * 100)}% Complete</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        session.status === 'running' ? 'bg-blue-500' :
                        session.status === 'paused' ? 'bg-yellow-500' :
                        session.status === 'completed' ? 'bg-green-500' : 'bg-gray-400'
                      }`}
                      style={{ width: `${Math.min(Math.max(session.progress || 0, 0) * 100, 100)}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              {session.status && (
                <div className="flex gap-2">
                  {session.status === 'running' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => handlePauseSession(e, session.id)}
                      className="text-yellow-600 border-yellow-300 hover:bg-yellow-50"
                    >
                      <Pause className="h-3 w-3 mr-1" /> Pause
                    </Button>
                  )}

                  {session.status === 'paused' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => handleResumeSession(e, session.id)}
                      className="text-green-600 border-green-300 hover:bg-green-50"
                    >
                      <Play className="h-3 w-3 mr-1" /> Resume
                    </Button>
                  )}

                  {(session.status === 'running' || session.status === 'paused') && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => handleStopSession(e, session.id)}
                      className="text-red-600 border-red-300 hover:bg-red-50"
                    >
                      <Square className="h-3 w-3 mr-1" /> Stop
                    </Button>
                  )}

                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectSession(session.id);
                    }}
                    className="ml-auto"
                  >
                    <Settings className="h-3 w-3 mr-1" /> Details
                  </Button>
                </div>
              )}
            </div>

            {/* Expanded Details */}
            {showDetails === session.id && (
              <div className="border-t border-gray-200 p-4 bg-gray-50">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Session Details</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Session ID:</span>
                        <span className="font-mono text-xs">{session.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Started:</span>
                        <span>{new Date(session.date).toLocaleString()}</span>
                      </div>
                      {session.startTime && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Start Time:</span>
                          <span>{session.startTime}</span>
                        </div>
                      )}
                      {session.estimatedCompletion && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Est. Completion:</span>
                          <span>{session.estimatedCompletion}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Performance Metrics</h4>
                    <div className="space-y-2 text-sm">
                      {session.bestStrategy && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Best Strategy:</span>
                          <span className="font-medium">{session.bestStrategy}</span>
                        </div>
                      )}
                      {session.capital && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Capital:</span>
                          <span>${session.capital.toLocaleString()}</span>
                        </div>
                      )}
                      {session.pnl !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">P&L:</span>
                          <span className={session.pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                            ${session.pnl >= 0 ? '+' : ''}{session.pnl.toFixed(2)}
                          </span>
                        </div>
                      )}
                      {session.riskLevel && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Risk Level:</span>
                          <Badge variant={session.riskLevel === 'LOW' ? 'neutral' :
                                         session.riskLevel === 'MEDIUM' ? 'warning' : 'bear'}>
                            {session.riskLevel}
                          </Badge>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {session.symbols && session.symbols.length > 0 && (
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-900 mb-2">Trading Symbols</h4>
                    <div className="flex flex-wrap gap-2">
                      {session.symbols.map((symbol, index) => (
                        <Badge key={index} variant="outline">
                          {symbol}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredAndSortedSessions.length === 0 && sessions.length > 0 && (
        <div className="text-center py-8">
          <Filter className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <div className="text-gray-500">No sessions match your filters</div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              setSearchTerm('');
              setStatusFilter('all');
            }}
            className="mt-3"
          >
            Clear Filters
          </Button>
        </div>
      )}
    </div>
  );
};

export default ActiveSessionsList;
