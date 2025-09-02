import React from 'react';
import { RefreshCw, Clock, BarChart2, Play, Pause } from 'lucide-react';
import { StatusBadge } from '@/components/ui/StatusBadge';
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
    <div className="space-y-3">
      {sessions.map((session) => (
        <div 
          key={session.id}
          className={`p-3 border rounded-md transition-colors cursor-pointer ${
            activeSessionId === session.id 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-200 hover:bg-gray-50'
          }`}
          onClick={() => onSelectSession(session.id)}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <BarChart2 className="h-4 w-4 mr-2 text-blue-500" />
              <div className="font-medium">{session.id.substring(0, 8)}...</div>
            </div>
            {session.status && (
              <StatusBadge 
                variant={session.status === 'running' ? 'bull' : 
                         session.status === 'paused' ? 'warning' : 
                         session.status === 'completed' ? 'neutral' : 'bear'}
              >
                {session.status}
              </StatusBadge>
            )}
          </div>
          
          <div className="grid grid-cols-2 gap-2 mb-2">
            <div className="text-xs text-gray-500 flex items-center">
              <Clock className="h-3 w-3 mr-1" />
              {formatElapsedTime(session.date)}
            </div>
            <div className="text-xs text-gray-500 flex items-center justify-end">
              Fitness: <span className="font-medium text-blue-600 ml-1">{session.bestFitness.toFixed(4)}</span>
            </div>
          </div>
          
          {(session.progress !== undefined && session.currentGeneration !== undefined && session.totalGenerations !== undefined) && (
            <div className="mt-2">
              <div className="flex justify-between items-center text-xs text-gray-500 mb-1">
                <span>Gen {session.currentGeneration}/{session.totalGenerations}</span>
                <span>{Math.round(session.progress * 100)}%</span>
              </div>
              <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`${styles.progressBarFill} ${styles[`progress${Math.round(Math.min(Math.max(session.progress || 0, 0) * 100, 100) / 5) * 5}`]}`}
                ></div>
              </div>
            </div>
          )}
          
          {session.status && (
            <div className="mt-3 flex justify-end">
              {session.status === 'running' ? (
                <button
                  className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200 flex items-center"
                  onClick={(e) => handlePauseSession(e, session.id)}
                >
                  <Pause className="h-3 w-3 mr-1" /> Pause
                </button>
              ) : session.status === 'paused' ? (
                <button
                  className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-md hover:bg-green-200 flex items-center"
                  onClick={(e) => handleResumeSession(e, session.id)}
                >
                  <Play className="h-3 w-3 mr-1" /> Resume
                </button>
              ) : null}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default ActiveSessionsList;
