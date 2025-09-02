import React, { useState } from 'react';
import CandidateCard from './CandidateCard';
import type { Candidate } from '@/types/candidate';
import { useOpportunities, useProbeOpportunity, usePaperOrderOpportunity } from '@/hooks/useOpportunities';

interface FilterState {
  edgeMin: number;
  spreadMax: number;
  advMin: number;
  timeframe: string;
  hideHeld: boolean;
  riskOn: boolean;
}

const OpportunityScanner: React.FC = () => {
  const [filters, setFilters] = useState<FilterState>({
    edgeMin: 0.15,
    spreadMax: 80,
    advMin: 1000000,
    timeframe: 'ALL',
    hideHeld: false,
    riskOn: true,
  });
  
  const { data: candidates, isLoading, error } = useOpportunities(filters);
  const probeOpportunity = useProbeOpportunity();
  const paperOrderOpportunity = usePaperOrderOpportunity();

  const handleProbe = async (id: string) => {
    try {
      await probeOpportunity(id);
      console.log(`Starting probe for opportunity ${id}`);
    } catch (error) {
      console.error('Error starting probe:', error);
    }
  };

  const handleTrade = async (id: string) => {
    try {
      await paperOrderOpportunity(id);
      console.log(`Submitting paper order for opportunity ${id}`);
    } catch (error) {
      console.error('Error submitting paper order:', error);
    }
  };

  return (
    <div className="container py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Opportunity Scanner</h1>
      </div>
      
      {/* Filters */}
      <div className="bg-neutral-900/60 p-4 rounded-lg mb-6">
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <label htmlFor="edgeMin" className="text-sm">Edge≥</label>
            <input
              id="edgeMin"
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="w-16 px-2 py-1 rounded bg-neutral-800 border border-neutral-700"
              value={filters.edgeMin}
              onChange={e => setFilters(prev => ({ ...prev, edgeMin: parseFloat(e.target.value) }))}
            />
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="spreadMax" className="text-sm">Spread≤</label>
            <input
              id="spreadMax"
              type="number"
              step="1"
              min="0"
              className="w-16 px-2 py-1 rounded bg-neutral-800 border border-neutral-700"
              value={filters.spreadMax}
              onChange={e => setFilters(prev => ({ ...prev, spreadMax: parseFloat(e.target.value) }))}
            />
            <span className="text-sm">bps</span>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="advMin" className="text-sm">ADV≥</label>
            <input
              id="advMin"
              type="number"
              step="100000"
              min="0"
              className="w-24 px-2 py-1 rounded bg-neutral-800 border border-neutral-700"
              value={filters.advMin}
              onChange={e => setFilters(prev => ({ ...prev, advMin: parseFloat(e.target.value) }))}
            />
          </div>
          <div className="flex items-center gap-2">
            <button
              className={`px-3 py-1 rounded ${filters.timeframe === 'ALL' ? 'bg-primary text-white' : 'bg-neutral-800'}`}
              onClick={() => setFilters(prev => ({ ...prev, timeframe: 'ALL' }))}
            >
              ALL
            </button>
            <button
              className={`px-3 py-1 rounded ${filters.timeframe === 'ST' ? 'bg-primary text-white' : 'bg-neutral-800'}`}
              onClick={() => setFilters(prev => ({ ...prev, timeframe: 'ST' }))}
            >
              ST
            </button>
            <button
              className={`px-3 py-1 rounded ${filters.timeframe === 'MT' ? 'bg-primary text-white' : 'bg-neutral-800'}`}
              onClick={() => setFilters(prev => ({ ...prev, timeframe: 'MT' }))}
            >
              MT
            </button>
          </div>
          <div className="flex items-center gap-4">
            <label className="inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={filters.hideHeld}
                onChange={e => setFilters(prev => ({ ...prev, hideHeld: e.target.checked }))}
              />
              <div className="relative w-11 h-6 bg-neutral-700 rounded-full peer peer-checked:bg-primary peer-focus:ring-1 peer-focus:ring-primary">
                <div className={`absolute w-4 h-4 bg-white rounded-full top-1 transition-all ${filters.hideHeld ? 'left-6' : 'left-1'}`}></div>
              </div>
              <span className="ml-2 text-sm">Hide held</span>
            </label>
            <label className="inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={filters.riskOn}
                onChange={e => setFilters(prev => ({ ...prev, riskOn: e.target.checked }))}
              />
              <div className="relative w-11 h-6 bg-neutral-700 rounded-full peer peer-checked:bg-primary peer-focus:ring-1 peer-focus:ring-primary">
                <div className={`absolute w-4 h-4 bg-white rounded-full top-1 transition-all ${filters.riskOn ? 'left-6' : 'left-1'}`}></div>
              </div>
              <span className="ml-2 text-sm">Risk-on</span>
            </label>
          </div>
        </div>
      </div>
      
      {/* Diamonds Section */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-3 flex items-center">
          <span className="text-yellow-400 mr-2">⭐</span> Diamonds
        </h2>
        <div className="auto-grid">
          {candidates?.filter(c => c.decision === 'PASS').slice(0, 3).map(c => (
            <CandidateCard
              key={c.id}
              c={c}
              onProbe={handleProbe}
              onTrade={handleTrade}
            />
          ))}
          {(!candidates || candidates.filter(c => c.decision === 'PASS').length === 0) && (
            <div className="col-span-full text-center py-8 bg-neutral-900/40 rounded-lg border border-neutral-800">
              <p className="text-muted-foreground">No diamond opportunities found at the moment</p>
            </div>
          )}
        </div>
      </div>
      
      {/* All Opportunities */}
      <div>
        <h2 className="text-xl font-semibold mb-3">All Opportunities</h2>
        {isLoading ? (
          <div className="text-center py-8">
            <p className="text-muted-foreground">Loading opportunities...</p>
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <p className="text-red-500">Error loading opportunities</p>
          </div>
        ) : (
          <div className="auto-grid">
            {candidates?.map(c => (
              <CandidateCard
                key={c.id}
                c={c}
                onProbe={handleProbe}
                onTrade={handleTrade}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default OpportunityScanner;
