# DEPRECATED ENTRY POINTS

⚠️ **These entry points are deprecated and will be removed in a future version.**

## Migration Guide

All functionality has been consolidated into the unified CLI:

```bash
# Instead of: python main.py
benbot evaluator --strategy momentum --symbols SPY,AAPL

# Instead of: python run_bot.py
benbot live --dry-run

# Instead of: python run_dashboard.py
benbot dashboard --port 8050

# Instead of: python cli.py backtest
benbot backtest --start-date 2024-01-01 --end-date 2024-12-31
```

## What Changed

- **Single Entry Point**: All operations now go through `benbot` command
- **Consistent Interface**: All commands use the same argument patterns
- **Better Documentation**: Comprehensive help with `benbot --help`
- **Maintainability**: Easier to maintain and extend

## Files Moved Here

- `main.py` → Use `benbot evaluator` instead
- `app.py` → Use `benbot live` instead
- `cli.py` → Use `benbot` subcommands instead
- `run_bot.py` → Use `benbot live` instead
- `run_dashboard.py` → Use `benbot dashboard` instead

## Next Steps

1. Update any scripts or automation to use the new CLI
2. Test all functionality with the new commands
3. Remove references to these deprecated files

For questions or issues, please refer to the main documentation or create an issue.
