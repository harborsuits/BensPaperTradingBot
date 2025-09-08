import { useState, Suspense, lazy } from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

// Lazy load complex components to avoid import issues
const EvoTesterDashboard = lazy(() => import('@/components/evotester/EvoTesterDashboard'));
const EvoTesterPanel = lazy(() => import('@/components/evotester/EvoTesterPanel'));
const AIBotCompetition = lazy(() => import('@/components/evotester/AIBotCompetition'));

/**
 * EvoTesterPage - Main component for the evolutionary testing page
 * Provides access to both the modern EvoTesterDashboard and legacy EvoTesterPanel interfaces
 */
function EvoTesterPage(): JSX.Element {
  const [activeView, setActiveView] = useState<'modern' | 'classic' | 'competition'>('modern');

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Evolutionary Strategy Tester</h1>
        <Tabs
          value={activeView}
          onValueChange={(value) => setActiveView(value as 'modern' | 'classic' | 'competition')}
          className="w-auto"
        >
          <TabsList>
            <TabsTrigger value="modern">Evolution Lab</TabsTrigger>
            <TabsTrigger value="classic">Classic View</TabsTrigger>
            <TabsTrigger value="competition">Bot Competition</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <Suspense fallback={
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading EvoTester...</p>
          </div>
        </div>
      }>
        {activeView === 'modern' ? (
          <EvoTesterDashboard />
        ) : activeView === 'competition' ? (
          <AIBotCompetition />
        ) : (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Legacy EvoTester Interface</CardTitle>
              </CardHeader>
              <CardContent>
                <EvoTesterPanel
                  onComplete={() => {
                    console.log('Evolution completed successfully!');
                  }}
                />
              </CardContent>
            </Card>
          </div>
        )}
      </Suspense>
    </div>
  );
}

export default EvoTesterPage;
