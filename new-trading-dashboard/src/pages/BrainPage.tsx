import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import BrainFlow from '@/components/BrainFlow';
import BrainScoringActivity from '@/components/BrainScoringActivity';
import BrainEvoFlow from '@/components/BrainEvoFlow';
import { Badge } from '@/components/ui/Badge';
import { Brain, Target, TrendingUp, Activity } from 'lucide-react';

export default function BrainPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('SPY');

  return (
    <div className="w-full py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-white">Brain Systems</h1>
            <p className="text-sm text-gray-400 mt-1">
              Autonomous, safe, evidence-first trading intelligence
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-white border-blue-500">
            Live Pilot Active
          </Badge>
          <Badge variant="outline" className="text-white border-purple-500">
            R&D Offline
          </Badge>
        </div>
      </div>

      {/* Four-Area Navigation */}
      <Tabs defaultValue="flow" className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-800">
          <TabsTrigger
            value="flow"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white flex items-center gap-2"
          >
            <Activity className="w-4 h-4" />
            Brain Flow
          </TabsTrigger>
          <TabsTrigger
            value="scoring"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white flex items-center gap-2"
          >
            <Target className="w-4 h-4" />
            Scoring Activity
          </TabsTrigger>
          <TabsTrigger
            value="decisions"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white flex items-center gap-2"
          >
            <TrendingUp className="w-4 h-4" />
            Trade Decisions
          </TabsTrigger>
          <TabsTrigger
            value="brain-evo"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white flex items-center gap-2"
          >
            <Brain className="w-4 h-4" />
            Brain + EvoFlow
          </TabsTrigger>
        </TabsList>

        {/* Brain Flow Tab */}
        <TabsContent value="flow" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">Brain Flow</h2>
                <p className="text-sm text-gray-400">
                  Per-symbol pipeline diagnostics - see where the autonomous loop stops and why
                </p>
              </div>
            </div>
            <BrainFlow />
          </div>
        </TabsContent>

        {/* Scoring Activity Tab */}
        <TabsContent value="scoring" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">Brain Scoring Activity</h2>
                <p className="text-sm text-gray-400">
                  Candidate ranking explainer - exactly why the winner won
                </p>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Symbol:</label>
                <select
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                  className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-white text-sm"
                >
                  <option value="SPY">SPY</option>
                  <option value="QQQ">QQQ</option>
                  <option value="AAPL">AAPL</option>
                  <option value="TSLA">TSLA</option>
                </select>
              </div>
            </div>
            <BrainScoringActivity symbol={selectedSymbol} />
          </div>
        </TabsContent>

        {/* Trade Decisions Tab */}
        <TabsContent value="decisions" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">Trade Decisions</h2>
                <p className="text-sm text-gray-400">
                  The bridge from ideas to trades - what will/would be traded
                </p>
              </div>
            </div>
            {/* We'll embed the TradeDecisionsPage content here */}
            <div className="border rounded-2xl p-4 bg-gray-900/50">
              <p className="text-center text-gray-400 py-8">
                Trade Decisions component integrated in /decisions route
              </p>
              <p className="text-center text-xs text-gray-500 mt-2">
                Visit the Decisions page to see Proposals, Trade Intents, and Executions tabs
              </p>
            </div>
          </div>
        </TabsContent>

        {/* Brain + EvoFlow Tab */}
        <TabsContent value="brain-evo" className="mt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">Brain + EvoFlow</h2>
                <p className="text-sm text-gray-400">
                  Live pilot + Offline R&D working in tandem, safely
                </p>
              </div>
            </div>
            <BrainEvoFlow />
          </div>
        </TabsContent>
      </Tabs>

      {/* Quick Status Summary */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="border rounded-lg p-4 bg-card">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium">Brain Flow</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Pipeline diagnostics per symbol
          </p>
        </div>

        <div className="border rounded-lg p-4 bg-card">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium">Scoring Activity</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Candidate ranking explainer
          </p>
        </div>

        <div className="border rounded-lg p-4 bg-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-purple-600" />
            <span className="text-sm font-medium">Trade Decisions</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Bridge from ideas to trades
          </p>
        </div>

        <div className="border rounded-lg p-4 bg-card">
          <div className="flex items-center gap-2 mb-2">
            <Brain className="w-4 h-4 text-yellow-600" />
            <span className="text-sm font-medium">Brain + EvoFlow</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Live pilot + Offline R&D
          </p>
        </div>
      </div>
    </div>
  );
}
