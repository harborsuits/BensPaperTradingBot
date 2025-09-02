import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import EvoTesterDashboard from '@/components/evotester/EvoTesterDashboard';
import EvoTesterPanel from '@/components/evotester/EvoTesterPanel';
import useEvoTesterWebSocket from '@/hooks/useEvoTesterWebSocket';
import { showSuccessToast } from '@/utils/toast.js';

/**
 * EvoTesterPage - Main component for the evolutionary testing page
 * Provides access to both the modern EvoTesterDashboard and legacy EvoTesterPanel interfaces
 */
function EvoTesterPage(): JSX.Element {
  const [activeView, setActiveView] = useState<'modern' | 'classic'>('modern');
  
  // Setup WebSocket updates for global notifications
  useEvoTesterWebSocket();

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Evolutionary Strategy Tester</h1>
        <Tabs 
          value={activeView} 
          onValueChange={(value) => setActiveView(value as 'modern' | 'classic')}
          className="w-auto"
        >
          <TabsList>
            <TabsTrigger value="modern">Modern View</TabsTrigger>
            <TabsTrigger value="classic">Classic View</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      
      {activeView === 'modern' ? (
        <EvoTesterDashboard />
      ) : (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Legacy EvoTester Interface</CardTitle>
            </CardHeader>
            <CardContent>
              <EvoTesterPanel 
                onComplete={() => {
                  showSuccessToast('Evolution completed successfully!');
                }}
              />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default EvoTesterPage;
