import React, { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Network, Activity, Database, Terminal, Cpu } from 'lucide-react';
import WebSocketMonitor from '@/components/developer/WebSocketMonitor';
import EventSystemMonitor from '@/components/developer/EventSystemMonitor';
import StrategyPrioritization from '@/components/strategy/StrategyPrioritization';

/**
 * Developer page with tools for debugging and monitoring the application
 */
const DeveloperPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('websocket');

  return (
    <div className="container py-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Developer Tools</h1>
        <div className="flex items-center space-x-2">
          <span className="px-2 py-1 bg-amber-500 text-white text-xs font-medium rounded-md">
            Development Mode
          </span>
        </div>
      </div>

      <Tabs 
        defaultValue="websocket" 
        value={activeTab} 
        onValueChange={setActiveTab}
        className="w-full"
      >
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="websocket">
            <Network size={16} className="mr-2" />
            WebSocket Monitor
          </TabsTrigger>
          <TabsTrigger value="eventSystem">
            <Activity size={16} className="mr-2" />
            Event System
          </TabsTrigger>
          <TabsTrigger value="dataExplorer">
            <Database size={16} className="mr-2" />
            Data Explorer
          </TabsTrigger>
          <TabsTrigger value="systemMetrics">
            <Cpu size={16} className="mr-2" />
            System Metrics
          </TabsTrigger>
          <TabsTrigger value="console">
            <Terminal size={16} className="mr-2" />
            Console
          </TabsTrigger>
        </TabsList>
        
        <div className="mt-6">
          <TabsContent value="websocket" className="mt-0">
            <WebSocketMonitor />
          </TabsContent>
          
          <TabsContent value="eventSystem" className="mt-0">
            <EventSystemMonitor />
          </TabsContent>
          
          <TabsContent value="dataExplorer" className="mt-0">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Database size={18} className="mr-2" />
                  Data Explorer
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Data explorer will be implemented in a future update.
                </p>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="systemMetrics" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <StrategyPrioritization />
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Cpu size={18} className="mr-2" />
                    System Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    System resource monitoring will be implemented in a future update.
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="console" className="mt-0">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Terminal size={18} className="mr-2" />
                  Developer Console
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Console output will be implemented in a future update.
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};

export default DeveloperPage;
