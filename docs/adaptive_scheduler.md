# Adaptive Market Context Scheduler

The Adaptive Market Context Scheduler is a dynamic scheduling system that analyzes market conditions and news at different frequencies during and outside market hours.

## Overview

The scheduler adapts its refresh rate based on market hours:

- **During Market Hours** (5:00 AM - 4:00 PM): Updates every 15 minutes
- **Outside Market Hours** (4:00 PM - 5:00 AM): Updates every 60 minutes
- **Daily Context Generation**: Special run at 5:00 AM to set daily strategy bias

## Key Features

- **Time-Aware Scheduling**: Dynamically adjusts update frequency based on market hours
- **Historical Data Storage**: Maintains history of context analysis for later review
- **API Integration**: Exposes control and monitoring endpoints through Flask API
- **Configurable Parameters**: Easily adjust time ranges and frequencies

## Installation

### Prerequisites

- Python 3.7+
- Required packages: `requests`, `beautifulsoup4`, `openai`, `schedule`, `python-dotenv`, `flask`

### Setup

1. Install required packages:
   ```bash
   pip install requests beautifulsoup4 openai schedule python-dotenv flask
   ```

2. Configure your `.env` file with API keys:
   ```
   MARKETAUX_API_KEY=your_marketaux_key
   OPENAI_API_KEY=your_openai_key
   MARKET_HOURS_START=05:00
   MARKET_HOURS_END=16:00
   MARKET_HOURS_INTERVAL=15
   AFTER_HOURS_INTERVAL=60
   CONTEXT_OUTPUT_DIR=data/market_context
   ```

## Running the Scheduler

### As a Standalone Service

Run the scheduler script directly:

```bash
python scripts/run_adaptive_context_scheduler.py --run-now
```

Command-line options:
- `--market-start`: Market hours start time (default: "05:00")
- `--market-end`: Market hours end time (default: "16:00")
- `--market-interval`: Update interval during market hours in minutes (default: 15)
- `--after-interval`: Update interval after market hours in minutes (default: 60)
- `--output-dir`: Output directory for context files (default: "data/market_context")
- `--run-now`: Run update immediately then start scheduler

### As a Systemd Service

1. Copy the service file to systemd:
   ```bash
   sudo cp deployment/adaptive-context-scheduler.service /etc/systemd/system/
   ```

2. Reload systemd, enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable adaptive-context-scheduler
   sudo systemctl start adaptive-context-scheduler
   ```

3. Check service status:
   ```bash
   sudo systemctl status adaptive-context-scheduler
   ```

### Integrated with Flask App

The scheduler is automatically started when running the Flask application unless disabled:

```bash
python -m trading_bot.app
```

## API Endpoints

When running with the Flask app, the following endpoints are available:

- `GET /api/context-scheduler/status`: Get current status of the scheduler
- `POST /api/context-scheduler/update`: Manually trigger a context update
   - Query param `daily=true` to run as a daily update
- `GET /api/context-scheduler/config`: Get current scheduler configuration
- `POST /api/context-scheduler/config`: Update scheduler configuration
   - JSON body with `market_hours_interval` and/or `after_hours_interval`

## Output Files

The scheduler generates these files:

- `data/market_context/current_market_context.json`: Latest context analysis
- `data/market_context/daily_strategy_bias.json`: Daily strategy bias (5 AM run)
- `data/market_context/history/context_YYYY-MM-DD_HHMM.json`: Historical context snapshots
- `data/market_context/history/strategy_bias_YYYY-MM-DD.json`: Historical strategy bias

## Monitoring and Troubleshooting

View logs:
```bash
tail -f logs/adaptive_context.log
```

Systemd service logs:
```bash
journalctl -u adaptive-context-scheduler -f
```

## Examples

### Manually update context via API

```bash
curl -X POST http://localhost:5000/api/context-scheduler/update
```

### Change update frequency via API

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"market_hours_interval": 30, "after_hours_interval": 120}' \
  http://localhost:5000/api/context-scheduler/config
``` 