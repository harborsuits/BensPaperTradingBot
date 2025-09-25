# BenBot Deployment & Remote Access Guide

## Current Setup
Your bot is running on your Mac with:
- **Backend**: Port 4000 (API & trading logic)
- **Frontend**: Port 3003 (UI dashboard)
- **Process Manager**: PM2 (handles restarts & monitoring)

## Network Resilience
✅ **Continues running without internet** - Bot stays active, pauses trading
✅ **Auto-recovers from disconnections** - Resumes when connection returns
✅ **Handles API failures** - Backoff & retry logic built-in
✅ **Prevents bad trades** - Won't trade on stale data

## Remote Access Options

### 1. Quick Status Check (from anywhere)
```bash
# SSH into your Mac (if enabled)
ssh yourusername@your-mac-ip
cd ~/Desktop/benbot
./check-bot-health.sh
```

### 2. View Logs Remotely
```bash
# Recent activity
pm2 logs benbot-backend --lines 50

# Real-time monitoring
pm2 monit
```

### 3. Emergency Controls
```bash
# Stop trading (bot stays running)
pm2 stop benbot-backend

# Restart if needed
pm2 restart benbot-backend

# Full shutdown
pm2 stop all
```

## Making It Bulletproof

### Enable Auto-Start on Boot
```bash
# Run this once to enable
./setup-autostart.sh
```

### Set Up Remote Access (Optional)
1. Enable Remote Login on your Mac:
   - System Settings → General → Sharing → Remote Login
   
2. Use Tailscale for secure access:
   ```bash
   # Install Tailscale
   brew install tailscale
   tailscale up
   ```

3. Access from anywhere:
   - SSH: `ssh username@your-tailscale-ip`
   - Web UI: `http://your-tailscale-ip:3003`

### Monitor from Your Phone
1. Install Tailscale on your phone
2. Browse to `http://your-tailscale-ip:3003`
3. View real-time portfolio & trades

## Data Persistence
- Trade history: Saved to SQLite database
- Positions: Synced with Tradier
- Logs: Stored in `live-api/logs/`
- PM2 processes: Saved in `~/.pm2/`

## Uptime Expectations
- Process crashes: Auto-restart (max 10 times)
- Network loss: Continues running, pauses trading
- Power loss: Will restart if auto-start enabled
- API limits: Smart backoff prevents bans

Your bot is designed to run 24/7 with minimal intervention! 🚀
