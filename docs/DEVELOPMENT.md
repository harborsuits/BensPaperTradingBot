# Development Guide

## Getting Started

### Prerequisites
- Node.js 18+ (for frontend)
- Python 3.9+ (for backend)
- Docker & Docker Compose (optional)

### First Time Setup

1. **Clone and navigate:**
   ```bash
   cd ~/Desktop/benbot
   ```

2. **Setup Python environment:**
   ```bash
   cd api
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Setup Node.js environment:**
   ```bash
   cd ../app
   pnpm install  # or npm install
   ```

## Development Workflow

### Running the Application

```bash
# Terminal 1: Backend API
cd api && source .venv/bin/activate
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend Dashboard
cd app
pnpm dev --port 5176

# Terminal 3: Database (if needed)
mongod  # or docker run mongo
```

### Key URLs
- **Frontend**: http://localhost:5176
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database**: mongodb://localhost:27017

## Project Structure

```
benbot/
├── app/                    # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Route pages
│   │   ├── hooks/          # React hooks
│   │   └── services/       # API clients
│   ├── public/             # Static assets
│   └── package.json
├── api/                    # Backend services
│   ├── trading_bot/        # Python trading engine
│   ├── live-api/          # Node.js API server
│   ├── requirements.txt
│   └── server.js
├── config/                 # Configuration files
├── scripts/                # Automation scripts
├── tests/                  # Test suites
└── docs/                   # Documentation
```

## Adding New Features

### Frontend Components
```bash
cd app/src/components
# Create new component file
# Import and use in pages
```

### Backend API Endpoints
```bash
cd api
# Add to server.js for Node.js endpoints
# Or to trading_bot/api/ for Python endpoints
```

### Trading Strategies
```bash
cd api/trading_bot/strategies
# Add new strategy class
# Register in strategy registry
```

## Testing

```bash
# Run all tests
cd tests && python -m pytest

# Run specific tests
python -m pytest tests/test_trading_engine.py -v

# Frontend tests
cd app && pnpm test
```

## Deployment

### Local Docker
```bash
cd infra
docker-compose up -d
```

### Production
See `docs/DEPLOYMENT.md` for production setup.

## Common Issues

### Python Issues
```bash
# Clear cache and reinstall
cd api && rm -rf __pycache__ .venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Node.js Issues
```bash
# Clear cache and reinstall
cd app && rm -rf node_modules .next
pnpm install  # or npm install
```

### Database Issues
```bash
# Reset database
mongo tradingbot --eval "db.dropDatabase()"
```

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Make changes and test
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/name`
5. Create Pull Request

See `docs/CONTRIBUTING.md` for detailed guidelines.
