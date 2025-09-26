module.exports = {
  apps: [{
    name: 'benbot-api',
    script: './minimal_server.js',
    cwd: '/home/ubuntu/benbot/live-api',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    
    // Environment variables
    env: {
      NODE_ENV: 'production',
      PORT: 4000,
      
      // Trading Configuration
      QUOTES_PROVIDER: 'tradier',
      AUTOLOOP_ENABLED: '1',
      STRATEGIES_ENABLED: '1',
      AI_ORCHESTRATOR_ENABLED: '1',
      AUTO_EVOLUTION_ENABLED: '1',
      
      // Tradier Configuration (Paper Trading)
      TRADIER_BASE_URL: 'https://sandbox.tradier.com/v1',
      // IMPORTANT: Set these in the .env file on the server!
      // TRADIER_API_KEY: 'your_key_here',
      // TRADIER_ACCOUNT_ID: 'your_account_here',
      
      // API Keys - Set in .env file!
      // FINNHUB_API_KEY: 'your_key_here',
      // MARKETAUX_API_TOKEN: 'your_key_here',
      
      // Crypto Configuration
      CRYPTO_ENABLED: '1',
      CRYPTO_SYMBOLS: 'BTC,ETH,SOL,MATIC,DOGE',
      
      // Capital Management
      PAPER_CAP_MAX: 100000,
      MAX_CAPITAL_PER_TRADE: 1000,
      MAX_OPEN_TRADES: 20,
      MAX_DAILY_TRADES: 100,
      
      // Performance
      TOURNAMENT_INTERVAL_MS: 300000, // 5 minutes
      AUTOLOOP_INTERVAL_MS: 15000,    // 15 seconds - faster reaction for news/stops
      
      // Logging
      LOG_LEVEL: 'info',
      LOG_TO_FILE: '1'
    },
    
    // Error handling
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
    log_file: './logs/pm2-combined.log',
    time: true,
    
    // Cron restart schedule (optional - restart daily at 3 AM)
    cron_restart: '0 3 * * *'
  }]
};