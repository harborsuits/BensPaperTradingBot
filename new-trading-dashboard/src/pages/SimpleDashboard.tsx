import React from 'react';

export default function SimpleDashboard() {
  return (
    <div style={{ padding: '20px' }}>
      <h1>BenBot Dashboard (Fallback)</h1>
      <p>If you're seeing this, the main dashboard has an error.</p>
      
      <h2>Debug Info:</h2>
      <ul>
        <li>API URL: {import.meta.env.VITE_API_BASE || '/api'}</li>
        <li>Environment: {import.meta.env.MODE}</li>
        <li>Time: {new Date().toISOString()}</li>
      </ul>
      
      <h2>Quick Links:</h2>
      <ul>
        <li><a href="/portfolio">Portfolio</a></li>
        <li><a href="/decisions">Decisions</a></li>
        <li><a href="/market">Market</a></li>
      </ul>
    </div>
  );
}
