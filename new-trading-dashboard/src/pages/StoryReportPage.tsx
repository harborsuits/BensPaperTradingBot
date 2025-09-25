import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { RefreshCw, Download, BookOpen } from 'lucide-react';
import { get } from '@/lib/api';

interface StorySection {
  title: string;
  story: string;
  whatThisMeans: string;
}

interface StoryReport {
  title: string;
  generated: string;
  sections: StorySection[];
  quickNumbers: {
    grade: string;
    bestDecision: string;
    worstDecision: string;
    luckFactor: number;
    smartFactor: number;
  };
}

export default function StoryReportPage() {
  const { data: report, isLoading, refetch } = useQuery<StoryReport>({
    queryKey: ['story-report'],
    queryFn: () => get('/api/report/story'),
    refetchInterval: 60000 // Refresh every minute
  });

  const downloadReport = () => {
    if (!report) return;
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trading-story-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <BookOpen className="h-12 w-12 mx-auto mb-4 text-blue-500 animate-pulse" />
          <p className="text-lg">Creating your trading story...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">📖 Today's Trading Story</h1>
          <p className="text-muted-foreground">
            {report?.generated || 'No report available yet'}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => refetch()} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={downloadReport} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
      </div>

      {/* Quick Report Card */}
      {report?.quickNumbers && (
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
          <CardHeader>
            <CardTitle>📊 Quick Report Card</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold">{report.quickNumbers.grade}</div>
                <div className="text-sm text-muted-foreground">Grade</div>
              </div>
              <div className="text-center">
                <div className="text-2xl">🎯 {report.quickNumbers.smartFactor}/10</div>
                <div className="text-sm text-muted-foreground">Smart Factor</div>
              </div>
              <div className="text-center">
                <div className="text-2xl">🍀 {report.quickNumbers.luckFactor}/10</div>
                <div className="text-sm text-muted-foreground">Luck Factor</div>
              </div>
              <div className="col-span-2">
                <div className="text-sm">
                  <div className="font-semibold text-green-600">
                    ✅ Best: {report.quickNumbers.bestDecision}
                  </div>
                  <div className="font-semibold text-red-600">
                    ❌ Worst: {report.quickNumbers.worstDecision}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Story Sections */}
      <ScrollArea className="h-[calc(100vh-300px)]">
        <div className="space-y-6">
          {report?.sections.map((section, index) => (
            <Card key={index} className="overflow-hidden">
              <CardHeader className="bg-gradient-to-r from-gray-50 to-gray-100">
                <CardTitle className="text-xl">{section.title}</CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                {/* Main Story */}
                <div className="prose prose-lg max-w-none mb-6">
                  <div 
                    className="whitespace-pre-wrap text-gray-700"
                    dangerouslySetInnerHTML={{ 
                      __html: section.story
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\n/g, '<br>')
                        .replace(/([0-9]+\.)/g, '<br>$1')
                    }}
                  />
                </div>

                {/* Developer Explanation */}
                <details className="bg-blue-50 rounded-lg p-4">
                  <summary className="cursor-pointer font-semibold text-blue-900 hover:text-blue-700">
                    🔧 How the Code Works (Click to expand)
                  </summary>
                  <div className="mt-3 text-sm text-blue-800 font-mono">
                    <div 
                      dangerouslySetInnerHTML={{ 
                        __html: section.whatThisMeans
                          .replace(/\n/g, '<br>')
                          .replace(/([a-zA-Z_]+\.js)/g, '<code class="bg-blue-100 px-1 py-0.5 rounded">$1</code>')
                          .replace(/([a-zA-Z_]+\(\))/g, '<code class="bg-blue-100 px-1 py-0.5 rounded">$1</code>')
                      }}
                    />
                  </div>
                </details>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>

      {/* No Report Message */}
      {!report && (
        <Card className="text-center py-12">
          <CardContent>
            <BookOpen className="h-16 w-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-xl font-semibold mb-2">No Story Yet!</h3>
            <p className="text-muted-foreground">
              The trading story will appear here after market hours.
              Check back after 4:30 PM EST!
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
