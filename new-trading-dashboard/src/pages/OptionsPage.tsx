import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import OptionsMetrics from '@/components/options/OptionsMetrics';
import OptionsTable from '@/components/options/OptionsTable';
import OptionsChainViewer from '@/components/options/OptionsChainViewer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Activity, TrendingUp, BookOpen, Settings } from 'lucide-react';

export default function OptionsPage() {
  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Options Trading</h1>
          <p className="text-gray-500 mt-1">Manage covered calls, cash-secured puts, and spreads</p>
        </div>
        <div className="flex gap-2">
          <span className="text-sm text-gray-500">Options allocation: 15% of capital</span>
        </div>
      </div>

      {/* Options Metrics Dashboard */}
      {/* <OptionsMetrics /> */}

      {/* Tabbed Content */}
      <Tabs defaultValue="positions" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="positions" className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Positions
          </TabsTrigger>
          <TabsTrigger value="chain" className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            Options Chain
          </TabsTrigger>
          <TabsTrigger value="strategies" className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Strategies
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        {/* Positions Tab */}
        <TabsContent value="positions" className="space-y-4">
          {/* <OptionsTable /> */}
          <p>Options positions will be displayed here</p>
        </TabsContent>

        {/* Options Chain Tab */}
        <TabsContent value="chain" className="space-y-4">
          {/* <OptionsChainViewer /> */}
          <p>Options chain will be displayed here</p>
        </TabsContent>

        {/* Strategies Tab */}
        <TabsContent value="strategies" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Covered Calls Strategy Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Covered Calls Strategy
                  <span className="text-sm text-green-600">Active</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Capital Allocated</span>
                    <span className="font-medium">$20,000</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Target Premium</span>
                    <span className="font-medium">2% monthly</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Delta Range</span>
                    <span className="font-medium">0.25 - 0.40</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">DTE Target</span>
                    <span className="font-medium">21-45 days</span>
                  </div>
                </div>
                <div className="pt-4 border-t">
                  <h4 className="text-sm font-medium mb-2">Recent Performance</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>This Month</span>
                      <span className="text-green-600">+$450 (2.25%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Win Rate</span>
                      <span>80% (4/5)</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Cash Secured Puts Strategy Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Cash Secured Puts Strategy
                  <span className="text-sm text-green-600">Active</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Capital Allocated</span>
                    <span className="font-medium">$15,000</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Target Premium</span>
                    <span className="font-medium">1.5% monthly</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Delta Range</span>
                    <span className="font-medium">0.20 - 0.35</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Strike Target</span>
                    <span className="font-medium">3-8% OTM</span>
                  </div>
                </div>
                <div className="pt-4 border-t">
                  <h4 className="text-sm font-medium mb-2">Recent Performance</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>This Month</span>
                      <span className="text-green-600">+$325 (2.17%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Win Rate</span>
                      <span>67% (2/3)</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Strategy Opportunities */}
          <Card>
            <CardHeader>
              <CardTitle>Opportunities</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <p className="font-medium">SPY 455 Call - Dec 20</p>
                    <p className="text-sm text-gray-500">IV: 18%, Premium: $2.40, Delta: 0.35</p>
                  </div>
                  <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                    Analyze
                  </button>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <p className="font-medium">QQQ 380 Put - Dec 20</p>
                    <p className="text-sm text-gray-500">IV: 22%, Premium: $3.90, Delta: -0.25</p>
                  </div>
                  <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                    Analyze
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Options Trading Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h4 className="text-sm font-medium mb-3">Risk Parameters</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Max Contracts per Trade</label>
                    <input type="number" defaultValue={5} className="w-20 px-2 py-1 border rounded" />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Min Days to Expiration</label>
                    <input type="number" defaultValue={21} className="w-20 px-2 py-1 border rounded" />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Max Capital Allocation</label>
                    <input type="number" defaultValue={15} className="w-20 px-2 py-1 border rounded" />
                    <span className="text-sm text-gray-500">%</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-3">Approved Symbols</h4>
                <div className="flex flex-wrap gap-2">
                  {['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'AMD', 'NVDA'].map(symbol => (
                    <span key={symbol} className="px-3 py-1 bg-gray-100 rounded text-sm">
                      {symbol}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-3">Strategy Preferences</h4>
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" defaultChecked />
                    <span className="text-sm">Enable Covered Calls</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" defaultChecked />
                    <span className="text-sm">Enable Cash Secured Puts</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" />
                    <span className="text-sm">Enable Vertical Spreads</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" />
                    <span className="text-sm">Auto-roll at 21 DTE</span>
                  </label>
                </div>
              </div>

              <div className="pt-4">
                <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                  Save Settings
                </button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
