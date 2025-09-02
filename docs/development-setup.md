# Development Setup

This guide will help you set up a development environment for the BensBot Trading System.

## Prerequisites

- Python 3.8+ (3.10+ recommended)
- Git
- pip
- virtualenv or conda (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/TheClitCommander/BensBot.git
cd BensBot
```

### 2. Create a Virtual Environment

Using virtualenv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using conda:

```bash
conda create -n bensbot python=3.10
conda activate bensbot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your API keys and configuration:

```bash
# Broker credentials
TRADIER_API_KEY=your_tradier_api_key
TRADIER_ACCOUNT_ID=your_tradier_account_id
TRADIER_SANDBOX=true

# Risk parameters
MAX_RISK_PCT=0.01
MAX_POSITION_PCT=0.05
INITIAL_CAPITAL=100000.0

# Market data APIs
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
FINNHUB_KEY=your_finnhub_key

# News APIs (for dashboard)
NEWSDATA_KEY=your_newsdata_key
MARKETAUX_KEY=your_marketaux_key

# Notifications
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# AI models (optional)
OPENAI_API_KEY=your_openai_key
```

### 5. Initialize Configuration Files

Run the configuration initializer:

```bash
python -m trading_bot.scripts.initialize_config
```

### 6. Run Tests

Verify that your setup works by running the core tests:

```bash
python -m pytest tests/unit
```

## Development Workflow

### Running the Trading Bot

Start the main orchestrator:

```bash
python -m trading_bot.core.main_orchestrator
```

### Running the API Server

Start the FastAPI server:

```bash
python -m trading_bot.api.app
```

The API will be available at http://localhost:8000, with documentation at http://localhost:8000/docs.

### Running the Dashboard

Start the dashboard web application:

```bash
cd dashboard
npm install  # First time only
npm start
```

The dashboard will be available at http://localhost:3000.

## Code Organization

```
trading_bot/
├── api/                # API layer (FastAPI)
├── auth/               # Authentication services
├── backtesting/        # Backtesting engine
├── brokers/            # Broker adapters
├── config/             # Configuration management
├── core/               # Core orchestration
├── dashboard/          # Web dashboard (React)
├── data/               # Data management
├── indicators/         # Technical indicators
├── models/             # ML models
├── notification/       # Notification services
├── risk/               # Risk management
├── scripts/            # Utility scripts
├── strategies/         # Trading strategies
└── tests/              # Test suite
```

## IDE Setup

### VSCode

Recommended extensions:
- Python
- Pylance
- Python Test Explorer
- GitLens
- autoDocstring

Recommended settings (`.vscode/settings.json`):

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.nosetestsEnabled": false,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

### PyCharm

- Open the project directory
- Set the project interpreter to your virtual environment
- Install the recommended plugins:
  - Mypy
  - Black Formatter
  - Requirements

## Style Guidelines

- Follow PEP 8
- Use type hints
- Document with docstrings (Google style)
- Keep line length to 100 characters
- Use meaningful variable names

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them with:

```bash
pip install pre-commit
pre-commit install
```

## Troubleshooting

### API Key Issues

If you encounter "Unauthorized" errors when using market data:
1. Check that your API keys are correctly set in `.env`
2. Verify the keys are valid by making a direct API request
3. Check API usage limits for your keys

### Dependency Issues

If you encounter import errors:
1. Make sure your virtual environment is activated
2. Update your dependencies: `pip install -r requirements.txt --upgrade`
3. Check for conflicting packages: `pip check`

### Test Failures

If tests are failing:
1. Run with higher verbosity: `pytest -v`
2. Isolate failing tests: `pytest tests/path/to/test.py::test_function -v`
3. Check environment setup and required API keys for tests
