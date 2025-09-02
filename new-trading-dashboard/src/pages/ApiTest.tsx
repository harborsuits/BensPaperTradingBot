import React, { useEffect, useState } from 'react';

interface TestResponse {
  success: boolean;
  data: any;
}

const ApiTest: React.FC = () => {
  const [regimeData, setRegimeData] = useState<any>(null);
  const [featuresData, setFeaturesData] = useState<any>(null);
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const testApis = async () => {
      try {
        // Test MSW interception of regime endpoint
        const regimeResponse = await fetch('/api/context/regime');
        const regimeJson = await regimeResponse.json();
        setRegimeData(regimeJson);

        // Test MSW interception of features endpoint
        const featuresResponse = await fetch('/api/context/features');
        const featuresJson = await featuresResponse.json();
        setFeaturesData(featuresJson);

        // Test MSW interception of sentiment endpoint
        const sentimentResponse = await fetch('/api/context/sentiment');
        const sentimentJson = await sentimentResponse.json();
        setSentimentData(sentimentJson);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    testApis();
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">MSW API Test</h1>
      
      {error && (
        <div className="p-4 bg-red-100 text-red-800 rounded mb-4">
          Error: {error}
        </div>
      )}

      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-2">Regime API Response:</h2>
          <pre className="bg-gray-100 p-3 rounded overflow-auto">
            {regimeData ? JSON.stringify(regimeData, null, 2) : 'Loading...'}
          </pre>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-2">Features API Response:</h2>
          <pre className="bg-gray-100 p-3 rounded overflow-auto">
            {featuresData ? JSON.stringify(featuresData, null, 2) : 'Loading...'}
          </pre>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-2">Sentiment API Response:</h2>
          <pre className="bg-gray-100 p-3 rounded overflow-auto">
            {sentimentData ? JSON.stringify(sentimentData, null, 2) : 'Loading...'}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default ApiTest;
