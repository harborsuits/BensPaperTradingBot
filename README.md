# BenBot - Advanced Trading System

A comprehensive, enterprise-grade algorithmic trading platform with AI-powered market analysis, risk management, and multi-broker support.

## 🏗️ Architecture Overview

```
benbot/
├── app/               # React/TypeScript frontend dashboard
├── api/               # Python trading engine + Node.js API server
├── services/          # Background services and workers
├── libs/              # Shared libraries and utilities
├── infra/             # Docker, Kubernetes, deployment configs
├── scripts/           # Automation and maintenance scripts
├── docs/              # Documentation and guides
├── config/            # Configuration files and settings
├── tests/             # Test suites and fixtures
└── tools/             # Development and utility tools
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/pnpm
- Python 3.9+
- Docker & Docker Compose (optional)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <your-repo-url> benbot
   cd benbot
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Install dependencies:**
   ```bash
   # Frontend
   cd app && pnpm install

   # Backend API
   cd ../api && python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Start the system:**
   ```bash
   # Terminal 1: Backend API (Python/Node.js)
   cd api && python -m uvicorn main:app --reload --port 8000

   # Terminal 2: Frontend Dashboard
   cd app && pnpm dev --port 5176
   ```

5. **Open in browser:**
   - Frontend: http://localhost:5176
   - API Docs: http://localhost:8000/docs

## 📊 Features

### 🤖 AI-Powered Trading
- **Market Regime Detection**: ML models identify bullish/bearish/sideways markets
- **Sentiment Analysis**: News and social media sentiment integration
- **Strategy Optimization**: Automated parameter tuning with genetic algorithms
- **Risk Management**: Dynamic position sizing and stop-loss management

### 📈 Multi-Asset Support
- **Stocks**: US equities with technical analysis
- **Options**: Complex spreads (iron condors, butterflies, etc.)
- **Forex**: Currency pairs with algorithmic strategies
- **Crypto**: Digital assets via Coinbase integration

### 🏢 Multi-Broker Support
- **Tradier**: Stocks and options trading
- **Alpaca**: Commission-free API trading
- **Coinbase**: Cryptocurrency trading
- **Interactive Brokers**: Professional-grade execution

### 🎯 Advanced Strategies
- **Technical Analysis**: 50+ indicators and pattern recognition
- **Machine Learning**: Ensemble models for prediction
- **Options Strategies**: Automated spread management
- **Arbitrage**: Statistical arbitrage across assets
- **Scalping**: High-frequency algorithmic trading

## 🛠️ Development

### Project Structure Details

#### `app/` - Frontend Dashboard
- React 18 with TypeScript
- Tailwind CSS for styling
- Real-time WebSocket connections
- Interactive charts and analytics

#### `api/` - Backend Services
- **Python Engine**: Core trading logic and ML models
- **Node.js API**: REST/WebSocket endpoints
- **Database**: MongoDB for trade storage
- **Message Queue**: Redis for real-time processing

#### `services/` - Background Workers
- Market data ingestion
- Strategy execution
- Risk monitoring
- Notification services

### Key Technologies
- **Frontend**: React, TypeScript, Tailwind CSS, Recharts
- **Backend**: Python (FastAPI), Node.js (Express), MongoDB
- **ML/AI**: TensorFlow, scikit-learn, pandas, numpy
- **Infrastructure**: Docker, Kubernetes, Redis, Nginx

## 🔧 Configuration

### Environment Variables
```bash
# API Keys (replace with your actual keys)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
TRADIER_API_KEY=your_tradier_key
COINBASE_API_KEY=your_coinbase_key

# Database
MONGODB_URL=mongodb://localhost:27017/tradingbot

# Application Settings
NODE_ENV=development
LOG_LEVEL=INFO
DEFAULT_BROKER=tradier
```

### Broker Setup
1. **Tradier**: Get API key from Tradier developer portal
2. **Alpaca**: Create account at alpaca.markets
3. **Coinbase**: Generate API keys in Coinbase Pro

## 🧪 Testing

```bash
# Run all tests
cd tests && python -m pytest

# Run specific test suite
python -m pytest tests/test_trading_engine.py

# Run with coverage
python -m pytest --cov=api --cov-report=html
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
cd infra && docker-compose up -d

# Or build individual services
docker build -t benbot-api ./api
docker build -t benbot-app ./app
```

### Production Setup
```bash
# Install system dependencies
sudo apt update && sudo apt install -y python3-dev mongodb redis-server

# Configure environment
export NODE_ENV=production
export MONGODB_URL=mongodb://localhost:27017/tradingbot

# Start services
systemctl start mongod redis
cd api && python -m uvicorn main:app --host 0.0.0.0 --port 8000
cd app && npm run build && npm run serve
```

## 📚 Documentation

- **[API Documentation](./docs/API.md)**: REST/WebSocket endpoints
- **[Strategy Guide](./docs/STRATEGIES.md)**: Available trading strategies
- **[Configuration](./docs/CONFIG.md)**: Detailed configuration options
- **[Deployment](./docs/DEPLOYMENT.md)**: Production deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for all React components
- Write tests for new features
- Update documentation for API changes

## 📄 License

This project is proprietary software. See LICENSE file for details.

## 🆘 Support

For support and questions:
- **Documentation**: Check the `/docs` folder first
- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub Discussions for questions

## 🔄 Recent Changes

### v2.0.0 (September 2025)
- ✅ **Major refactor**: Clean monorepo structure
- ✅ **React Dashboard**: Modern frontend with real-time updates
- ✅ **AI Integration**: Enhanced ML models for market prediction
- ✅ **Multi-broker**: Support for Tradier, Alpaca, Coinbase
- ✅ **Risk Management**: Advanced position sizing and stops
- ✅ **Docker Support**: Containerized deployment

---

**Built with ❤️ for serious traders**

*Note: This is a sophisticated trading system. Use paper trading first and understand the risks before deploying with real money.*
