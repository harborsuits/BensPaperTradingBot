import React from 'react';
import { initial, capitalize, seed } from '@/utils/string';
import { fmtNum, fmtPercent } from '@/utils/number';

// Test data with edge cases
const TEST_DATA = [
  { id: 1, name: null, assetClass: undefined, score: "12.3456", winRate: "0.577" },
  { id: 2, name: "", assetClass: "💹", score: "abc", winRate: null },
  { id: 3, name: "  spaced  ", assetClass: "equities", score: 0, winRate: 0 },
  { id: 4 }, // totally sparse
];

// Simple component to validate our utility functions
const SafetyTest: React.FC = () => {
  return (
    <div className="p-4 border rounded-md">
      <h1 className="text-xl font-bold mb-4">Safety Test</h1>
      
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b">
            <th className="text-left p-2">ID</th>
            <th className="text-left p-2">Name</th>
            <th className="text-left p-2">Initial</th>
            <th className="text-left p-2">Asset Class</th>
            <th className="text-left p-2">Score</th>
            <th className="text-left p-2">Win Rate</th>
            <th className="text-left p-2">Seed Hash</th>
          </tr>
        </thead>
        <tbody>
          {TEST_DATA.map(item => (
            <tr key={item.id || 'unknown'} className="border-b">
              <td className="p-2">{item.id || '—'}</td>
              <td className="p-2">{capitalize(item.name)}</td>
              <td className="p-2">{initial(item.name)}</td>
              <td className="p-2">{capitalize(item.assetClass)}</td>
              <td className="p-2">{fmtNum(item.score)}</td>
              <td className="p-2">{fmtPercent(item.winRate)}</td>
              <td className="p-2">{seed(item.name) || 0}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 className="text-lg font-bold mt-6 mb-2">Debug Values:</h2>
      <pre className="bg-gray-100 p-3 overflow-auto rounded text-xs">
        {JSON.stringify(TEST_DATA, null, 2)}
      </pre>
    </div>
  );
};

export default SafetyTest;
