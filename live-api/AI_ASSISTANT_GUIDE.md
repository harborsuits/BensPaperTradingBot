# AI Assistant Guide - BenBot Dashboard

## ⚠️ READ THIS BEFORE MODIFYING ANY CODE

### Current Working State
The dashboard is fully functional with the following fixes applied:
- ✅ Autopilot working (no getAllStrategies error)
- ✅ Position quantities displaying correctly (not NaN)
- ✅ Market prices updating (not stale)
- ✅ 16 strategies active and running

### Critical Files - DO NOT MODIFY WITHOUT VERIFICATION
1. `minimal_server.js` - Contains critical dashboard fixes
2. `minimal_server.js.fixed-backup` - Backup of working version

### Before Making Any Changes:
1. Run `./verify-dashboard-fixes.sh` to ensure fixes are intact
2. Check if dashboard is currently working at http://localhost:3003
3. If you need to modify minimal_server.js, create a new backup first

### Additional Fix Applied:
5. **AutoLoop StrategyManager Reference**
   - Root cause: autoLoop tries to access strategyManager but it's not set
   - Fix location: Line ~407 in minimal_server.js
   - Solution: Add `autoLoop.strategyManager = strategyManager;` after autoLoop creation

### Known Issues That Have Been Fixed:
1. **getAllStrategies Error**
   - Root cause: API returning array but code expecting object
   - Fix location: Line ~3879 in minimal_server.js
   
2. **NaN Position Data**
   - Root cause: Field name mismatch (qty vs quantity)
   - Fix location: Position transformation in /api/paper/positions endpoint

3. **Stale Market Prices**
   - Root cause: No mock provider when Tradier not configured
   - Fix location: Mock quotes provider in /api/quotes endpoint

### How to Start the Server:
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
./start-server.sh
```

### How to Test if Everything Works:
1. Check API health: `curl http://localhost:4000/api/health | jq`
2. Check strategies: `curl http://localhost:4000/api/strategies/active | jq`
3. Check positions: `curl http://localhost:4000/api/paper/positions | jq`
4. Visit dashboard: http://localhost:3003

### If You Break Something:
1. Run `./verify-dashboard-fixes.sh` to identify what's broken
2. Restore from backup: `cp minimal_server.js.fixed-backup minimal_server.js`
3. Restart using `./start-server.sh`

### DO NOT:
- Use sed/grep to bulk modify minimal_server.js
- Remove the mock providers
- Change field names in API responses without checking frontend
- Modify without understanding the data flow

### ALWAYS:
- Run verification script before and after changes
- Test the dashboard UI after API changes
- Create backups before major modifications
- Check that strategies are activated after server start

## Remember: The dashboard expects specific field names and data formats. Breaking these contracts will cause NaN values and errors in the UI.
