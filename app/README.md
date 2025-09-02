# BenBot Trading Dashboard

A modern React-based trading dashboard for monitoring and managing automated trading strategies. This dashboard connects to the BenBot FastAPI backend for real-time data and control of trading operations.

![BenBot Trading Dashboard](https://via.placeholder.com/800x400?text=BenBot+Trading+Dashboard)

## Features

- **Real-time Dashboard**: Monitor market conditions, portfolio performance, and active strategies
- **Portfolio Management**: View and manage positions for both live and paper trading accounts
- **Strategy Management**: Configure, activate, and monitor trading strategies
- **Trade Decision Analysis**: Analyze the scoring and execution of trade decisions
- **Market Context**: View market sentiment, trends, and news that influence trading
- **Evolutionary Strategy Tester**: Develop and optimize strategies through evolutionary algorithms
- **System Logs**: Monitor system logs and alerts for better operations management
- **WebSocket Integration**: Real-time updates for all critical data points

## Getting Started

### Prerequisites

- Node.js (v16+)
- npm (v8+) or yarn (v1.22+)
- BenBot FastAPI Backend running and accessible

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/benbot-trading-dashboard.git
   cd benbot-trading-dashboard
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Create an `.env` file in the root directory with the following variables:
   ```
   REACT_APP_API_URL=http://localhost:8000
   REACT_APP_WS_URL=ws://localhost:8000
   ```

4. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```

The application will be available at `http://localhost:3000`.

## Production Deployment

To build the application for production:

```bash
npm run build
# or
yarn build
```

This will create a `build` directory with optimized production files.

### Environment Variables for Production

For production deployment, ensure these environment variables are set:

- `REACT_APP_API_URL`: URL of your production FastAPI backend
- `REACT_APP_WS_URL`: WebSocket URL of your production FastAPI backend

## API Integration Points

The dashboard interacts with the FastAPI backend through these key endpoints:

### Authentication

- `POST /auth/login`: User authentication
- `GET /auth/user`: Get current user info

### Market Context

- `GET /context`: Get current market context data
- `WS /ws/context`: WebSocket for real-time market updates

### Portfolio

- `GET /portfolio/{account}`: Get portfolio data for specified account
- `GET /portfolio/{account}/positions`: Get current positions
- `GET /portfolio/{account}/trades`: Get trade history
- `WS /ws/portfolio`: WebSocket for real-time portfolio updates

### Strategies

- `GET /strategies`: Get all strategies
- `GET /strategies/active`: Get active strategies
- `POST /strategies/{id}/activate`: Activate a strategy
- `POST /strategies/{id}/deactivate`: Deactivate a strategy
- `PUT /strategies/{id}`: Update strategy configuration
- `WS /ws/strategies`: WebSocket for real-time strategy updates

### Trade Decisions

- `GET /decisions/latest`: Get latest trade decisions
- `GET /decisions/history`: Get trade decision history
- `GET /decisions/{id}`: Get specific trade decision details
- `WS /ws/decisions`: WebSocket for real-time decision updates

### EvoTester

- `GET /evotester/history`: Get run history
- `POST /evotester/run`: Start a new evolution run
- `POST /evotester/pause`: Pause current run
- `POST /evotester/resume`: Resume paused run
- `POST /evotester/stop`: Stop current run
- `POST /evotester/save/{id}`: Save a strategy
- `WS /ws/evotester`: WebSocket for real-time evolution updates

### Logs

- `GET /logs`: Get system logs
- `GET /alerts`: Get system alerts
- `WS /ws/logs`: WebSocket for real-time log updates

## Project Structure

```
benbot-trading-dashboard/
├── public/                 # Public assets
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── ui/             # Basic UI components
│   │   └── domain/         # Domain-specific components
│   ├── contexts/           # React contexts
│   ├── hooks/              # Custom React hooks
│   ├── pages/              # Page components
│   ├── services/           # API and WebSocket services
│   ├── types/              # TypeScript type definitions
│   ├── utils/              # Utility functions
│   ├── App.tsx             # Main application component
│   └── index.tsx           # Application entry point
└── package.json            # Dependencies and scripts
```

## Performance Optimization

For optimized performance, the application:

- Uses React Query for efficient data fetching and caching
- Implements virtualized lists for large datasets
- Debounces frequent WebSocket updates
- Uses memoization for expensive calculations
- Implements lazy loading for route components

## User Experience Considerations

- Responsive design works on desktop and tablet
- Toast notifications for user feedback
- Error states for network failures
- Loading states for asynchronous operations
- Dark mode support

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BenBot FastAPI Backend team
- React and Tailwind CSS contributors
