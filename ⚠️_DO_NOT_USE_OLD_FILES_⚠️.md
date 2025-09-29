# ⚠️ CRITICAL WARNING - DO NOT USE OLD FILES ⚠️

## 🚫 THESE DIRECTORIES CONTAIN BROKEN/OUTDATED CODE:
- `server-archive/` - OLD BROKEN CODE
- `backup/` - OUTDATED BACKUPS
- Any file with timestamps like `server-backup-1758429442.js`

## ✅ ONLY USE THESE FILES:
- `live-api/minimal_server.js` - Current working server
- `live-api/minimal_server.js.fixed-backup` - Known good backup
- `live-api/start-benbot.sh` - Proper startup script

## 🔧 TO START THE SYSTEM:
```bash
cd live-api
./start-benbot.sh
```

## 🆘 IF SOMETHING BREAKS:
```bash
cd live-api
cp minimal_server.js.fixed-backup minimal_server.js
./start-benbot.sh
```

## ⚠️ FOR AI ASSISTANTS:
READ `live-api/AI_ASSISTANT_INSTRUCTIONS.md` BEFORE MAKING ANY CHANGES!

---
THIS SYSTEM HANDLES REAL MONEY. BREAKING IT CAUSES FINANCIAL LOSS.
---
