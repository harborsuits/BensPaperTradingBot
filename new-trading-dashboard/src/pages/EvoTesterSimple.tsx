import React from 'react';

export default function EvoTesterSimple() {
  const [error, setError] = React.useState<any>(null);

  React.useEffect(() => {
    // Log that we made it here
    console.log('EvoTesterSimple mounted successfully');
  }, []);

  // Error boundary
  if (error) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold text-red-600 mb-4">Error in EvoTester</h1>
        <pre className="bg-red-100 p-4 rounded">
          {error.toString()}
          {error.stack && '\n\n' + error.stack}
        </pre>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">EvoTester (Simple Version)</h1>
      
      <div className="space-y-4">
        <div className="bg-blue-100 p-4 rounded">
          <p className="font-semibold">✅ This page is rendering correctly!</p>
        </div>

        <div className="bg-gray-100 p-4 rounded">
          <h2 className="font-semibold mb-2">Next Steps:</h2>
          <ol className="list-decimal list-inside space-y-1">
            <li>If you see this, the routing works fine</li>
            <li>The issue is likely in the lazy-loaded components</li>
            <li>Check browser console for specific errors</li>
            <li>Try the full EvoTester at <a href="/evotester" className="text-blue-600 underline">/evotester</a></li>
          </ol>
        </div>

        <div className="bg-yellow-100 p-4 rounded">
          <h2 className="font-semibold mb-2">Try Direct Component Test:</h2>
          <button 
            onClick={() => {
              import('@/components/evotester/EvoTesterDashboard')
                .then(module => {
                  console.log('Successfully loaded EvoTesterDashboard:', module);
                  alert('EvoTesterDashboard loaded successfully! Check console.');
                })
                .catch(err => {
                  console.error('Failed to load EvoTesterDashboard:', err);
                  setError(err);
                });
            }}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Test Load EvoTesterDashboard
          </button>
        </div>
      </div>
    </div>
  );
}
