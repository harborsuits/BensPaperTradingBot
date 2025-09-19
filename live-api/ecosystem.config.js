module.exports = {
  apps: [
    {
      name: 'live-api',
      cwd: __dirname,
      script: 'server.js',
      env: {
        PORT: 4000,
        NODE_ENV: 'production',
      },
      watch: false,
      max_memory_restart: '300M',
      out_file: 'logs/pm2-out.log',
      error_file: 'logs/pm2-err.log',
      merge_logs: true,
      kill_timeout: 5000,
      env_production: {
        NODE_ENV: 'production'
      }
    },
  ]
}
