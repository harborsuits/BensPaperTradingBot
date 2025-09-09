import { useState, useEffect, useCallback } from 'react';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { evoTesterApi } from '@/services/api';
import { fetchEvoGenerations } from '@/services/evotesterApi';

export type EvoProgress = {
  sessionId: string;
  running: boolean;
  currentGeneration: number;
  totalGenerations: number;
  startTime: string;
  estimatedCompletionTime?: string;
  elapsedTime?: string;
  progress: number;
  bestFitness: number;
  averageFitness: number;
  status: string;
  errorMessage?: string;
};

export type GenerationResult = {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  diversityScore: number;
  bestIndividual: Record<string, any>;
  elapsedTime: string;
  timestamp: string;
};

export type EvoResult = {
  sessionId: string;
  config: any;
  topStrategies: Array<Record<string, any>>;
  generationsData: GenerationResult[];
  startTime: string;
  endTime: string;
  totalRuntime: string;
  status: 'running' | 'completed' | 'failed';
  errorMessage?: string;
};

/**
 * Shared data synchronization utility for EvoTester components
 * Ensures all components stay in sync when data changes occur
 */
export const invalidateEvoTesterQueries = (queryClient: any) => {
  queryClient.invalidateQueries(['evoTester']);
  queryClient.invalidateQueries(['strategies']);
  queryClient.invalidateQueries(['portfolio']);
  queryClient.invalidateQueries(['context']);
  queryClient.invalidateQueries(['decisions']);
  queryClient.invalidateQueries(['trades']);
  queryClient.invalidateQueries(['pipeline']);
};

/**
 * React hook for EvoTester WebSocket updates.
 * Handles all aspects of real-time communication with the EvoTester backend.
 */
export function useEvoTesterUpdates(sessionId?: string) {
  // Connection status
  const { isConnected } = useWebSocketChannel('evotester');
  
  // State for progress and results
  const [progress, setProgress] = useState<EvoProgress | null>(null);
  const [generations, setGenerations] = useState<GenerationResult[]>([]);
  const [result, setResult] = useState<EvoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  // Reset state when session ID changes
  useEffect(() => {
    if (sessionId) {
      setGenerations([]);
      setResult(null);
      setError(null);
      
      // Fetch initial status
      const fetchStatus = async () => {
        try {
          const response = await evoTesterApi.getEvoStatus(sessionId);
          if (response.success && response.data) {
            setProgress(response.data);
            setIsRunning(response.data.running);
          }
        } catch (err) {
          console.error('Error fetching EvoTester status:', err);
        }
      };
      // Fetch historical generations to populate charts for completed sessions
      const fetchSeries = async () => {
        try {
          const series = await fetchEvoGenerations(sessionId);
          if (Array.isArray(series) && series.length) setGenerations(series as any);
        } catch (err) {
          // Non-fatal; just leave chart empty
        }
      };
      
      fetchStatus();
      fetchSeries();
    }
  }, [sessionId]);

  // Handle progress updates
  useWebSocketMessage<EvoProgress>(
    'evo_progress',
    (message) => {
      // Only process updates for the current session
      if (!sessionId || message.data.sessionId === sessionId) {
        setProgress(message.data);
        setIsRunning(message.data.running);
      }
    }
  );

  // Handle generation completion updates
  useWebSocketMessage<GenerationResult>(
    'evo_generation_complete',
    (message) => {
      if (!sessionId || (progress && progress.sessionId === sessionId)) {
        setGenerations(prev => [...prev, message.data]);
      }
    }
  );

  // Handle final result
  useWebSocketMessage<EvoResult>(
    'evo_complete',
    (message) => {
      if (!sessionId || message.data.sessionId === sessionId) {
        setResult(message.data);
        setIsRunning(false);
        
        // Fetch final result from API for completeness
        const fetchResult = async () => {
          try {
            const response = await evoTesterApi.getEvoResults(message.data.sessionId);
            if (response.success && response.data) {
              // Format the result as EvoResult
              const formattedResult: EvoResult = {
                sessionId: message.data.sessionId,
                config: message.data.config || {},
                topStrategies: response.data,
                generationsData: generations,
                startTime: message.data.startTime || new Date().toISOString(),
                endTime: message.data.endTime || new Date().toISOString(),
                totalRuntime: message.data.totalRuntime || 'N/A',
                status: message.data.status as 'completed' | 'running' | 'failed',
                errorMessage: message.data.errorMessage
              };
              setResult(formattedResult);
            }
          } catch (err) {
            console.error('Error fetching final EvoTester result:', err);
          }
        };
        
        fetchResult();
      }
    }
  );

  // Handle errors
  useWebSocketMessage<any>(
    'evo_error',
    (message) => {
      if (!sessionId || message.data.sessionId === sessionId) {
        setError(message.data.errorMessage || 'An unknown error occurred');
        setIsRunning(false);
      }
    }
  );

  // Function to start a new EvoTester run
  const startEvoTest = useCallback(async (config: any) => {
    try {
      const response = await evoTesterApi.startEvoTest(config);
      if (response.success && response.data) {
        // Accept either session_id or sessionId from backend
        const sid = (response.data as any).session_id || (response.data as any).sessionId;
        return sid || null;
      } else {
        setError(response.error || 'Failed to start EvoTester');
        return null;
      }
    } catch (err) {
      console.error('Error starting EvoTester:', err);
      setError('Failed to start EvoTester');
      return null;
    }
  }, []);

  // Function to stop an ongoing EvoTester run
  const stopEvoTest = useCallback(async (id: string) => {
    try {
      const response = await evoTesterApi.stopEvoTest(id);
      return response.success;
    } catch (err) {
      console.error('Error stopping EvoTester:', err);
      return false;
    }
  }, []);

  return {
    isConnected,
    isRunning,
    progress,
    generations,
    result,
    error,
    startEvoTest,
    stopEvoTest
  };
}

export default useEvoTesterUpdates;
