import React, { useState, useEffect } from 'react';
import { useEvoTesterUpdates } from '@/hooks/useEvoTesterUpdates';
import './EvoTesterPanel.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FiSettings, FiPlay, FiStopCircle, FiAlertTriangle, FiInfo, FiBarChart2 } from 'react-icons/fi';

interface EvoTesterPanelProps {
  defaultConfig?: any;
  onComplete?: (result: any) => void;
  className?: string;
}

const DEFAULT_CONFIG = {
  populationSize: 50,
  generations: 20,
  crossoverRate: 0.7,
  mutationRate: 0.2,
  elitismCount: 2,
  fitnessMetric: 'sharpe',
  timeframe: 'daily',
  symbols: ['SPY', 'QQQ', 'AAPL'],
  constraintsEnabled: true
};

const EvoTesterPanel: React.FC<EvoTesterPanelProps> = ({ 
  defaultConfig = DEFAULT_CONFIG, 
  onComplete,
  className = ''
}) => {
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [config, setConfig] = useState(defaultConfig);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  
  // Get real-time updates using our custom hook
  const { 
    isConnected, 
    isRunning, 
    progress, 
    generations, 
    result, 
    error, 
    startEvoTest, 
    stopEvoTest 
  } = useEvoTesterUpdates(sessionId);

  // Update chart data when new generation results arrive
  useEffect(() => {
    if (generations.length > 0) {
      const data = generations.map(gen => ({
        generation: gen.generation,
        bestFitness: parseFloat(gen.bestFitness.toFixed(4)),
        averageFitness: parseFloat(gen.averageFitness.toFixed(4)),
        diversity: parseFloat(gen.diversityScore.toFixed(4))
      }));
      setChartData(data);
    }
  }, [generations]);

  // When the test completes, call the onComplete callback
  useEffect(() => {
    if (result && onComplete) {
      onComplete(result);
    }
  }, [result, onComplete]);

  // Start a new EvoTester run
  const handleStart = async () => {
    const newSessionId = await startEvoTest(config);
    if (newSessionId) {
      setSessionId(newSessionId);
    }
  };

  // Stop the current EvoTester run
  const handleStop = async () => {
    if (sessionId) {
      await stopEvoTest(sessionId);
    }
  };

  // Convert progress to percentage
  const getProgressPercentage = () => {
    if (!progress) return 0;
    return progress.progress || 0;
  };

  return (
    <div className={`rounded-lg bg-navy-700 shadow-lg p-4 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-white">Evolutionary Strategy Tester</h2>
        <div className="flex space-x-2">
          {isConnected ? (
            <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
              Live
            </span>
          ) : (
            <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">
              Offline
            </span>
          )}
          <button 
            onClick={() => setIsConfigOpen(!isConfigOpen)}
            className="p-2 rounded-full hover:bg-navy-600"
            title="Configuration"
          >
            <FiSettings className="text-gray-300" />
          </button>
        </div>
      </div>

      {/* Configuration Panel */}
      {isConfigOpen && (
        <div className="mb-4 p-4 bg-navy-600 rounded-lg">
          <h3 className="text-sm font-medium text-white mb-3">Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Population Size</label>
              <input 
                type="number" 
                min={10}
                max={500}
                value={config.populationSize} 
                onChange={(e) => setConfig({...config, populationSize: parseInt(e.target.value)})}
                className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
                disabled={isRunning}
                id="population-size"
                aria-label="Population Size"
                title="Population Size"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1" htmlFor="generations">Generations</label>
              <input 
                type="number"
                min={5}
                max={200}
                value={config.generations} 
                onChange={(e) => setConfig({...config, generations: parseInt(e.target.value)})}
                className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
                disabled={isRunning}
                id="generations"
                aria-label="Generations"
                title="Generations"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1" htmlFor="mutation-rate">Mutation Rate</label>
              <input 
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={config.mutationRate} 
                onChange={(e) => setConfig({...config, mutationRate: parseFloat(e.target.value)})}
                className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
                disabled={isRunning}
                id="mutation-rate"
                aria-label="Mutation Rate"
                title="Mutation Rate"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1" htmlFor="crossover-rate">Crossover Rate</label>
              <input 
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={config.crossoverRate} 
                onChange={(e) => setConfig({...config, crossoverRate: parseFloat(e.target.value)})}
                className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
                disabled={isRunning}
                id="crossover-rate"
                aria-label="Crossover Rate"
                title="Crossover Rate"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1" htmlFor="fitness-metric">Fitness Metric</label>
              <select
                value={config.fitnessMetric}
                onChange={(e) => setConfig({...config, fitnessMetric: e.target.value})}
                className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
                disabled={isRunning}
                id="fitness-metric"
                aria-label="Fitness Metric"
                title="Fitness Metric"
              >
                <option value="sharpe">Sharpe Ratio</option>
                <option value="sortino">Sortino Ratio</option>
                <option value="winRate">Win Rate</option>
                <option value="profit">Net Profit</option>
                <option value="drawdown">Min Drawdown</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1" htmlFor="timeframe">Timeframe</label>
              <select
                value={config.timeframe}
                onChange={(e) => setConfig({...config, timeframe: e.target.value})}
                className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
                disabled={isRunning}
                id="timeframe"
                aria-label="Timeframe"
                title="Timeframe"
              >
                <option value="daily">Daily</option>
                <option value="hourly">Hourly</option>
                <option value="minute">Minute</option>
              </select>
            </div>
          </div>
          
          <div className="mt-4">
            <label className="text-xs text-gray-400 block mb-1" htmlFor="symbols">Symbols (comma separated)</label>
            <input 
              type="text"
              value={config.symbols.join(', ')}
              onChange={(e) => setConfig({...config, symbols: e.target.value.split(',').map(s => s.trim()).filter(Boolean)})}
              className="w-full bg-navy-800 border border-navy-500 rounded px-3 py-2 text-sm text-white"
              disabled={isRunning}
              placeholder="AAPL, MSFT, GOOG"
              id="symbols"
              aria-label="Symbols"
              title="Symbols"
            />
          </div>
          
          <div className="mt-4 flex items-center">
            <input 
              type="checkbox"
              id="constraintsEnabled"
              checked={config.constraintsEnabled}
              onChange={(e) => setConfig({...config, constraintsEnabled: e.target.checked})}
              className="mr-2"
              disabled={isRunning}
            />
            <label htmlFor="constraintsEnabled" className="text-xs text-gray-400">Enable Trading Constraints</label>
          </div>
        </div>
      )}

      {/* Progress and Controls */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center">
            <span className="text-sm text-white font-medium mr-2">Progress:</span>
            <span className="text-sm text-blue-400">
              {progress ? `${progress.currentGeneration} / ${progress.totalGenerations} generations` : 'Not started'}
            </span>
          </div>
          <div className="flex space-x-2">
            {!isRunning ? (
              <button
                onClick={handleStart}
                disabled={isRunning}
                className="flex items-center px-3 py-1.5 rounded text-sm font-medium bg-green-700 text-white hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <FiPlay className="mr-1.5" /> Start
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="flex items-center px-3 py-1.5 rounded text-sm font-medium bg-red-700 text-white hover:bg-red-600"
              >
                <FiStopCircle className="mr-1.5" /> Stop
              </button>
            )}
          </div>
        </div>
        
        {/* Progress bar */}
        <div className="w-full bg-navy-600 rounded-full h-2.5 overflow-hidden">
          <div 
            className={`progress-bar progress-${Math.round(getProgressPercentage() / 5) * 5}`}
          ></div>
        </div>
        
        {/* Status message */}
        {error && (
          <div className="mt-2 flex items-start text-red-500 text-sm">
            <FiAlertTriangle className="mr-1.5 mt-0.5 flex-shrink-0" /> {error}
          </div>
        )}
        
        {progress && progress.status && (
          <div className="mt-2 flex items-start text-gray-400 text-xs">
            <FiInfo className="mr-1.5 mt-0.5 flex-shrink-0" /> 
            Status: {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}
            {progress.status === 'running' && (
              <> • Best Fitness: {progress.bestFitness.toFixed(4)}</>
            )}
          </div>
        )}
      </div>

      {/* Fitness Chart */}
      {chartData.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-white mb-2">Fitness Evolution</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2D3748" />
                <XAxis 
                  dataKey="generation" 
                  stroke="#A0AEC0" 
                  fontSize={12}
                />
                <YAxis 
                  stroke="#A0AEC0" 
                  fontSize={12}
                  domain={['dataMin', 'dataMax']}
                />
                <Tooltip 
                  contentStyle={{ backdropFilter: 'blur(4px)' }} 
                  wrapperClassName="tooltip-dark" 
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="bestFitness" 
                  stroke="#3182CE" 
                  strokeWidth={2}
                  activeDot={{ r: 8 }} 
                  name="Best Fitness"
                />
                <Line 
                  type="monotone" 
                  dataKey="averageFitness" 
                  stroke="#805AD5" 
                  strokeWidth={2}
                  activeDot={{ r: 6 }} 
                  name="Average Fitness"
                />
                <Line 
                  type="monotone" 
                  dataKey="diversity" 
                  stroke="#38B2AC" 
                  strokeWidth={1.5}
                  activeDot={{ r: 5 }} 
                  name="Population Diversity"
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="mb-2">
          <h3 className="text-sm font-medium text-white mb-2 flex items-center">
            <FiBarChart2 className="mr-1.5" /> Evolution Results
          </h3>
          
          {result.topStrategies && result.topStrategies.length > 0 ? (
            <div className="space-y-3">
              {result.topStrategies.slice(0, 3).map((strategy, index) => (
                <div key={index} className="p-3 bg-navy-600 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <span className="text-white font-medium">
                        Strategy #{index + 1} 
                        {index === 0 && <span className="ml-1 text-yellow-400">★</span>}
                      </span>
                      <div className="text-sm text-gray-400">
                        Fitness: <span className="text-blue-400">{strategy.fitness?.toFixed(4) || 'N/A'}</span>
                      </div>
                    </div>
                    <button className="text-xs bg-blue-700 hover:bg-blue-600 text-white px-2 py-1 rounded">
                      View Details
                    </button>
                  </div>
                  
                  {strategy.parameters && (
                    <div className="mt-2">
                      <div className="text-xs text-gray-400 mb-1">Parameters:</div>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(strategy.parameters).slice(0, 6).map(([key, value]) => (
                          <div key={key} className="text-xs">
                            <span className="text-gray-500">{key}:</span>{' '}
                            <span className="text-white">{value?.toString() || 'N/A'}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-400 text-sm">No strategies found in results</div>
          )}
          
          <div className="mt-3 text-xs text-gray-400">
            Total Runtime: {result.totalRuntime || 'N/A'} • 
            Status: <span className={result.status === 'completed' ? 'text-green-400' : 'text-yellow-400'}>
              {result.status?.charAt(0).toUpperCase() + result.status?.slice(1) || 'Unknown'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvoTesterPanel;
