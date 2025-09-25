# 🖥️ Benbot Local Indefinite Running Guide

## Option 1: PM2 Process Manager (Recommended)

### Installation
```bash
# Install PM2
npm install -g pm2

# Start Benbot
cd ~/Desktop/benbot
pm2 start ecosystem.config.js

# View status
pm2 status

# View logs
pm2 logs

# Monitor in real-time
pm2 monit
```

### Auto-start on Mac boot
```bash
# Generate startup script
pm2 startup

# Copy and run the command it outputs (will require sudo)
# Example: sudo env PATH=$PATH:/usr/local/bin pm2 startup launchd -u bendickinson --hp /Users/bendickinson

# Save current process list
pm2 save
```

### Useful PM2 Commands
```bash
pm2 restart all         # Restart all processes
pm2 stop all           # Stop all processes
pm2 delete all         # Remove all processes
pm2 logs benbot-backend --lines 100  # View last 100 lines of backend logs
pm2 reload benbot-backend  # Zero-downtime reload
```

## Option 2: LaunchAgent (Mac Native)

### Create LaunchAgent for Backend
```bash
cat > ~/Library/LaunchAgents/com.benbot.backend.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.benbot.backend</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/node</string>
        <string>/Users/bendickinson/Desktop/benbot/live-api/minimal_server.js</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>TRADIER_API_KEY</key>
        <string>KU2iUnOZIUFre0wypgyOn8TgmGxI</string>
        <key>TRADIER_ACCOUNT_ID</key>
        <string>VA1201776</string>
        <key>FINNHUB_API_KEY</key>
        <string>cqg6r9hr01qj0vhhf6fgcqg6r9hr01qj0vhhf6g0</string>
        <key>MARKETAUX_API_TOKEN</key>
        <string>sG6o8FgyTvJ8VxXFyBMOFWhJJ81QzB</string>
        <key>PORT</key>
        <string>4000</string>
        <key>AUTOLOOP_ENABLED</key>
        <string>1</string>
        <key>STRATEGIES_ENABLED</key>
        <string>1</string>
        <key>AI_ORCHESTRATOR_ENABLED</key>
        <string>1</string>
        <key>AUTO_EVOLUTION_ENABLED</key>
        <string>1</string>
        <key>OPTIONS_ENABLED</key>
        <string>1</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>/Users/bendickinson/Desktop/benbot/live-api</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/bendickinson/Desktop/benbot/logs/backend.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/bendickinson/Desktop/benbot/logs/backend-error.log</string>
</dict>
</plist>
EOF

# Load the service
launchctl load ~/Library/LaunchAgents/com.benbot.backend.plist
```

## Option 3: Simple Bash Script
```bash
# Run the forever script
~/Desktop/benbot/scripts/run-forever.sh
```

## Monitoring & Maintenance

### Check if running
```bash
# PM2
pm2 status

# Manual check
ps aux | grep -E "node.*minimal_server|vite.*3003"

# Check ports
lsof -i :3003,4000
```

### View logs
```bash
# PM2 logs
pm2 logs

# Manual logs
tail -f ~/Desktop/benbot/logs/backend.log
tail -f ~/Desktop/benbot/logs/frontend.log
```

### Daily maintenance
```bash
# Restart to clear memory (PM2 does this automatically at 3 AM)
pm2 restart all

# Check disk space
df -h ~/Desktop/benbot

# Clean old logs (keeps last 7 days)
find ~/Desktop/benbot/logs -name "*.log" -mtime +7 -delete
```

## Troubleshooting

### If services won't start
1. Check ports aren't in use: `lsof -i :3003,4000`
2. Kill existing processes: `pkill -f "node minimal_server" && pkill -f vite`
3. Check logs for errors: `pm2 logs --lines 50`

### If Mac sleeps
1. System Preferences > Energy Saver
2. Check "Prevent computer from sleeping automatically when display is off"
3. Or use: `caffeinate -s` while running

### Performance issues
1. Check memory: `pm2 monit`
2. Restart services: `pm2 restart all`
3. Clear logs if too large: `pm2 flush`

## Best Practices

1. **Use PM2** - It's the most robust solution
2. **Monitor daily** - Check dashboard once a day
3. **Review logs** - Look for errors or warnings
4. **Backup data** - The `/data` folder contains trading history
5. **Update carefully** - Stop services before updating code

## Next Steps: Cloud Deployment

When ready for 24/7 operation without your Mac:
```bash
# Use the deployment script
~/Desktop/benbot/deployment/quick-deploy.sh <your-server-ip>
```

Recommended VPS providers:
- DigitalOcean: $5/month droplet
- Linode: $5/month Nanode
- AWS Lightsail: $3.50/month
- Vultr: $5/month
