import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Eye } from 'lucide-react';
import { j } from '@/lib/api';

interface EvidenceButtonProps {
  endpoint: string;
  title?: string;
}

export function EvidenceButton({ endpoint, title }: EvidenceButtonProps) {
  const [open, setOpen] = useState(false);
  const { data, refetch, isFetching } = useQuery({
    queryKey: ['evidence', endpoint],
    queryFn: () => j<any>(endpoint),
    enabled: open
  });

  const handleOpen = () => {
    setOpen(true);
    refetch();
  };

  return (
    <>
      <button
        onClick={handleOpen}
        className="inline-flex items-center gap-1 px-2 py-1 text-xs text-gray-400 hover:text-gray-200 border border-gray-600 hover:border-gray-500 rounded transition-colors"
        title="Show raw API response"
      >
        <Eye className="w-3 h-3" />
        Evidence
      </button>

      {open && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <div className="flex items-center gap-2">
                <h3 className="text-white font-medium">
                  {title || 'API Evidence'}
                </h3>
                <code className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
                  {endpoint}
                </code>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => refetch()}
                  disabled={isFetching}
                  className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded transition-colors"
                >
                  {isFetching ? 'Loading...' : 'Refresh'}
                </button>
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `evidence-${endpoint.replace(/[^a-zA-Z0-9]/g, '-')}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }}
                  disabled={!data}
                  className="px-3 py-1 text-sm bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded transition-colors"
                  title="Download JSON evidence"
                >
                  📥 Download
                </button>
                <button
                  onClick={() => setOpen(false)}
                  className="px-3 py-1 text-sm text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded transition-colors"
                >
                  Close
                </button>
              </div>
            </div>

            <div className="p-4 overflow-auto max-h-[60vh]">
              <pre className="text-xs text-gray-300 bg-gray-800 p-3 rounded border border-gray-700 overflow-auto">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
