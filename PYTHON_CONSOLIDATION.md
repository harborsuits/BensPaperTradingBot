# Python Entry Point Consolidation

## Overview

The BenBot system previously had **21 different main.py files** and **54 run*.py files** scattered throughout the codebase. This created massive confusion about which file to run for what purpose.

## New Structure

All Python functionality is now accessible through a single entry point:

```bash
python main.py [command] [options]
```

### Available Commands

1. **Server** (default) - Start the API server
   ```bash
   python main.py              # Starts live-api server by default
   python main.py server       # Explicit server command
   python main.py server --type simple   # Use simple_server.py
   python main.py server --type minimal  # Use minimal_server.py
   ```

2. **Bot** - Start the trading bot
   ```bash
   python main.py bot
   python main.py bot --config custom_config.yaml
   ```

3. **Backtest** - Run backtesting
   ```bash
   python main.py backtest SPY
   python main.py backtest AAPL --strategy news_momentum_v2
   ```

4. **Dashboard** - Start dashboard (deprecated)
   ```bash
   python main.py dashboard    # Legacy Python dashboard
   ```

## Migration Guide

### Old Way → New Way

| Old Command | New Command |
|-------------|-------------|
| `python simple_server.py` | `python main.py server --type simple` |
| `python trading_bot/orchestrator.py` | `python main.py bot` |
| `python trading_bot/backtest/run_backtest.py` | `python main.py backtest [symbol]` |
| `python run_bot.py` | `python main.py bot` |
| `python run_dashboard.py` | `python main.py dashboard` |

## Benefits

1. **Single Entry Point** - No more confusion about which file to run
2. **Clear Commands** - Explicit commands for different functions
3. **Consistent Interface** - Same pattern for all operations
4. **Easy Discovery** - `python main.py --help` shows all options
5. **Reduced Complexity** - From 75+ entry points to just 1

## Implementation Status

- ✅ Created unified `main.py` entry point
- ✅ Mapped all major functions
- ⚠️  Start scripts need updating to use new entry point
- ⚠️  Documentation needs updating
- ⚠️  Old entry points should be deprecated/removed

## Next Steps

1. Update all shell scripts to use `main.py`
2. Add deprecation warnings to old entry points
3. Update documentation
4. Eventually remove old entry points

## Note for Buyers

This consolidation significantly reduces the complexity of the system and makes it much easier to understand and maintain. The old entry points still exist for backwards compatibility but should be considered deprecated.
