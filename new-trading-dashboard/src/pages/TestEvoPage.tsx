import React from 'react';

export default function TestEvoPage() {
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [evoStatus, setEvoStatus] = React.useState<any>(null);

  React.useEffect(() => {
    // Test if API is working
    fetch('/api/evo/status')
      .then(res => res.json())
      .then(data => {
        setEvoStatus(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">EvoTester Debug Page</h1>
      
      <div className="space-y-4">
        <div className="bg-gray-100 p-4 rounded">
          <h2 className="font-semibold mb-2">Component Status:</h2>
          <p>✅ Page is rendering</p>
        </div>

        <div className="bg-gray-100 p-4 rounded">
          <h2 className="font-semibold mb-2">API Status:</h2>
          {loading && <p>Loading...</p>}
          {error && <p className="text-red-600">Error: {error}</p>}
          {evoStatus && (
            <pre className="text-xs overflow-auto">
              {JSON.stringify(evoStatus, null, 2)}
            </pre>
          )}
        </div>

        <div className="bg-gray-100 p-4 rounded">
          <h2 className="font-semibold mb-2">Troubleshooting:</h2>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>Check browser console for errors</li>
            <li>Check Network tab for failed requests</li>
            <li>Try hard refresh (Cmd+Shift+R)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
