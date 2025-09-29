# CRITICAL: AI Assistant Instructions for BenBot

## ⚠️ READ THIS FIRST - SYSTEM IS IN PRODUCTION

This trading system is actively running with real money at stake. Any breaking changes can cause financial losses. Follow these instructions EXACTLY.

## 🚫 FORBIDDEN ACTIONS - NEVER DO THESE

1. **NEVER** modify files from these directories:
   - `server-archive/`
   - `backup/`
   - Any file with a timestamp in the name (e.g., `server-backup-1758429442.js`)

2. **NEVER** revert these critical fixes:
   - Strategy API array iteration fix
   - Position data transformation (quantity/cost_basis)
   - Mock quotes provider
   - Rate limiter for quotes endpoint
   - BrainIntegrator enhanced recorder connection
   - PaperBroker real quotes integration
   - AutoLoop parallel evaluation

3. **NEVER** change these values without explicit permission:
   - Trading thresholds (currently in testing mode)
   - Rate limits
   - Port numbers (4000 for API, 3003 for dashboard)

## ✅ REQUIRED ACTIONS BEFORE ANY CHANGES

### 1. Run System Integrity Check
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
./verify-system-integrity.sh
```

If this fails, DO NOT PROCEED. Ask the user for guidance.

### 2. Check What's Running
```bash
# Check if servers are running
lsof -ti:4000  # Backend
lsof -ti:3003  # Frontend

# Check system status
curl -s http://localhost:4000/api/health | jq
```

### 3. Read Protected Files Manifest
Always read `/Users/bendickinson/Desktop/benbot/live-api/PROTECTED_FILES_MANIFEST.md` before making changes.

## 🛠️ SAFE PROCEDURES

### Starting the System
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
./start-benbot.sh
```

### Restarting After Issues
```bash
# Kill everything first
lsof -ti:4000 | xargs kill -9 2>/dev/null
lsof -ti:3003 | xargs kill -9 2>/dev/null

# Restore from backup if needed
cd /Users/bendickinson/Desktop/benbot/live-api
cp minimal_server.js.fixed-backup minimal_server.js

# Start fresh
./start-benbot.sh
```

### Making Code Changes

1. **ALWAYS** create a backup first:
   ```bash
   cp minimal_server.js minimal_server.js.backup-$(date +%s)
   ```

2. Make your changes

3. Verify integrity:
   ```bash
   ./verify-system-integrity.sh
   ```

4. If verification passes, update the fixed backup:
   ```bash
   cp minimal_server.js minimal_server.js.fixed-backup
   ```

## 📋 Current System State (as of September 29, 2025)

### What's Working
- Dashboard displays live data correctly
- Paper trading is functional
- Learning system is active with aggressive thresholds
- Parallel strategy evaluation is implemented
- Real-time price integration works

### Known Issues
- Old backup files exist in `server-archive/` - DO NOT USE THEM
- Some terminal logs show old server attempts - IGNORE THEM

### Active Features
- **Trading Mode**: Paper trading with aggressive thresholds
- **Learning**: Enhanced performance recorder with hot hand detection
- **Evaluation**: Parallel strategy evaluation for better performance
- **Safety**: Circuit breaker and rate limiting active

## 🎯 Common User Requests and Solutions

### "The dashboard is messed up"
1. Check if servers are running
2. Run `./start-benbot.sh`
3. Tell user to hard refresh browser (Cmd+Shift+R)

### "It's not trading"
1. Check AutoLoop status: `curl -s http://localhost:4000/api/autoloop/status | jq`
2. Check circuit breaker: `curl -s http://localhost:4000/api/circuit-breaker/status | jq`
3. If blocked, see `CIRCUIT_BREAKER_RESET.md`

### "Make it trade more aggressively"
- Thresholds are already set to testing mode (very aggressive)
- DO NOT lower them further without explicit permission

## 🚨 EMERGENCY PROCEDURES

### If Everything Breaks
```bash
cd /Users/bendickinson/Desktop/benbot/live-api

# Kill everything
lsof -ti:4000 | xargs kill -9 2>/dev/null
lsof -ti:3003 | xargs kill -9 2>/dev/null

# Restore from known good backup
cp minimal_server.js.fixed-backup minimal_server.js

# Start fresh
./start-benbot.sh
```

### If User Reports Financial Loss
1. IMMEDIATELY stop the system
2. Create a full backup of all logs
3. Do NOT make any changes without explicit permission
4. Ask user to provide specific timestamps and details

## 📝 BEFORE YOU LEAVE

Always:
1. Ensure system is running properly
2. Run integrity check one final time
3. Tell user the exact command to restart if needed: `cd /Users/bendickinson/Desktop/benbot/live-api && ./start-benbot.sh`
4. Remind user to hard refresh browser if dashboard looks wrong

## ⚠️ FINAL WARNING

This is a PRODUCTION TRADING SYSTEM. Every change you make can affect real money. When in doubt:
- ASK THE USER
- READ THE DOCUMENTATION
- RUN VERIFICATION SCRIPTS
- MAKE BACKUPS

The user trusts you to keep their trading system running. Don't break that trust.
