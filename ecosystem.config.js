const fs = require('fs');
const path = require('path');

// Read .env file manually
const envFile = path.join(__dirname, '.env');
const envVars = {};

if (fs.existsSync(envFile)) {
  const envContent = fs.readFileSync(envFile, 'utf8');
  envContent.split('\n').forEach(line => {
    if (line && !line.startsWith('#')) {
      const [key, value] = line.split('=');
      if (key && value) {
        envVars[key.trim()] = value.trim();
      }
    }
  });
}

module.exports = {
  apps: [{
    name: 'benbot-backend',
    script: './live-api/minimal_server.js',
    cwd: './',
    env: {
      NODE_ENV: 'development',
      PORT: 4000,
      STRATEGIES_ENABLED: '1',
      AI_ORCHESTRATOR_ENABLED: '1',
      ...envVars
    },
    error_file: './live-api/logs/backend-err.log',
    out_file: './live-api/logs/backend-out.log',
    merge_logs: true,
    max_restarts: 10,
    min_uptime: '10s',
    watch: false
  }]
};