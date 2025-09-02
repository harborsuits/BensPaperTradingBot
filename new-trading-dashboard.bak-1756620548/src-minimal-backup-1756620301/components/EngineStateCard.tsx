import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { formatDistanceToNow, formatDistance } from 'date-fns';
import { Switch } from '@headlessui/react';

interface EngineState {
  running: boolean;
  tick_id: string;
  last_tick_ts: number;
  next_tick_eta: number;
  tick_interval_seconds: number;
  total_ticks: number;
  uptime_seconds: number;
}

const fetchEngineState = async (): Promise<EngineState> => {
  const { data } = await axios.get('/api/engine/state');
  return data;
};

const toggleEngine = async (running: boolean): Promise<EngineState> => {
  const { data } = await axios.post('/api/engine/toggle', { running });
  return data;
};

const EngineStateCard: React.FC = () => {
  const queryClient = useQueryClient();
  
  const { data, isLoading, error } = useQuery<EngineState>({
    queryKey: ['engineState'],
    queryFn: fetchEngineState,
    refetchInterval: 5000,
    staleTime: 2000,
  });

  const mutation = useMutation({
    mutationFn: toggleEngine,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['engineState'] });
    },
  });

  const handleToggle = () => {
    if (data) {
      mutation.mutate(!data.running);
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const formatTimeDistance = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return formatDistanceToNow(date, { addSuffix: true });
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const formatNextTick = (timestamp: number) => {
    const now = Math.floor(Date.now() / 1000);
    const secondsRemaining = timestamp - now;
    
    if (secondsRemaining <= 0) {
      return 'Due now';
    }
    
    if (secondsRemaining < 60) {
      return `${secondsRemaining}s`;
    }
    
    const minutes = Math.floor(secondsRemaining / 60);
    const seconds = secondsRemaining % 60;
    return `${minutes}m ${seconds}s`;
  };

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-4 mb-4">
        <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Engine State</h2>
        <div className="animate-pulse mt-2">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-4 mb-4">
        <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Engine State</h2>
        <div className="text-red-500 mt-2">
          Error loading engine state
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-4 mb-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Engine State</h2>
        <div className="flex items-center">
          <span className="mr-2 text-sm text-gray-600 dark:text-gray-300">
            {data?.running ? 'Running' : 'Paused'}
          </span>
          <Switch
            checked={data?.running || false}
            onChange={handleToggle}
            className={`${
              data?.running ? 'bg-green-500' : 'bg-gray-400'
            } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
          >
            <span
              className={`${
                data?.running ? 'translate-x-6' : 'translate-x-1'
              } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
            />
          </Switch>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="text-gray-600 dark:text-gray-400">Last Tick:</div>
        <div className="text-gray-800 dark:text-gray-200 tabular-nums">
          {formatTimestamp(data?.last_tick_ts || 0)}
          <span className="text-xs text-gray-500 ml-1">
            ({formatTimeDistance(data?.last_tick_ts || 0)})
          </span>
        </div>
        
        <div className="text-gray-600 dark:text-gray-400">Next Tick:</div>
        <div className="text-gray-800 dark:text-gray-200 tabular-nums">
          {formatNextTick(data?.next_tick_eta || 0)}
        </div>
        
        <div className="text-gray-600 dark:text-gray-400">Tick ID:</div>
        <div className="text-gray-800 dark:text-gray-200 font-mono text-xs truncate">
          {data?.tick_id || 'N/A'}
        </div>
        
        <div className="text-gray-600 dark:text-gray-400">Interval:</div>
        <div className="text-gray-800 dark:text-gray-200 tabular-nums">
          {data?.tick_interval_seconds || 0}s
        </div>
        
        <div className="text-gray-600 dark:text-gray-400">Total Ticks:</div>
        <div className="text-gray-800 dark:text-gray-200 tabular-nums">
          {data?.total_ticks || 0}
        </div>
        
        <div className="text-gray-600 dark:text-gray-400">Uptime:</div>
        <div className="text-gray-800 dark:text-gray-200 tabular-nums">
          {formatUptime(data?.uptime_seconds || 0)}
        </div>
      </div>
      
      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={() => {
            axios.post('/api/engine/dev/force_tick')
              .then(() => queryClient.invalidateQueries({ queryKey: ['engineState'] }));
          }}
          className="text-xs bg-blue-500 hover:bg-blue-600 text-white py-1 px-2 rounded"
        >
          Force Tick
        </button>
      </div>
    </div>
  );
};

export default EngineStateCard;
