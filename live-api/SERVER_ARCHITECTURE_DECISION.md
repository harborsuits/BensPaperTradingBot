# Server Architecture Decision Required

## Current Situation

You have **7+ different server files** competing for the same job:

### File Analysis

| Server File | Lines | Status | Issues |
|------------|-------|---------|---------|
| **server.js** | 9,239 | Currently Running | - Missing paper trading fixes<br>- Tries Tradier first (causing wrong data)<br>- Too large to maintain |
| **minimal_server.js** | 4,081 | Has Fixes | - Has paper trading fixes<br>- Not currently running<br>- May be missing some endpoints |
| **simple_server.js** | 1,337 | Unknown | - Unclear purpose<br>- Not documented |
| **server_simple.js** | 495 | Unknown | - Different from simple_server.js<br>- Very basic |
| **opportunities-server.js** | 303 | Specialized | - Only handles opportunities |

## The Core Problem

Your fixes are scattered across different servers:
- Paper trading fixes → `minimal_server.js`
- Evolution endpoints → `server.js`
- Running server → `server.js` (but broken)

**This is why nothing works consistently!**

## Recommendation: Use ONE Server

### Option 1: Consolidate on `minimal_server.js` ✅ (Recommended)

**Pros:**
- Already has paper trading fixes
- Half the size (4k vs 9k lines)
- Easier to maintain
- Working implementation

**Cons:**
- Need to port missing endpoints from server.js
- Need to test everything

**How to switch:**
```bash
pm2 stop benbot-backend
pm2 start ecosystem.config.js --only benbot-backend --env production -- minimal_server.js
```

### Option 2: Fix `server.js`

**Pros:**
- Already running
- Has all features

**Cons:**
- 9,000+ lines (unmaintainable)
- Need to re-apply all fixes
- Will break again

### Option 3: Create New Clean Server

**Pros:**
- Clean architecture
- Best long-term solution

**Cons:**
- Time consuming
- Risk breaking everything

## Immediate Action Needed

1. **Choose one server** (I recommend minimal_server.js)
2. **Delete or archive the others**
3. **Port any missing features to the chosen server**
4. **Never create competing servers again**

## Why This Matters

- Your positions show wrong prices because server.js uses Tradier data
- Evolution endpoints 404 because they're not in the running server
- Fixes don't work because they're in the wrong file
- You can't debug because you don't know which server to look at

## Decision Time

What would you like to do?
1. Switch to minimal_server.js (I can help port missing features)
2. Fix server.js (not recommended)
3. Continue with the mess (really not recommended)

**This is the most important architectural decision for your trading system.**
