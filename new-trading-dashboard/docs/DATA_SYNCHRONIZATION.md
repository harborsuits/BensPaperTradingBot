# Data Synchronization Architecture

## Overview

The trading dashboard uses a centralized data synchronization system to ensure all components display consistent, real-time data from your Tradier account.

## Key Components

### 1. DataSyncContext (`/src/contexts/DataSyncContext.tsx`)
The central hub for all data synchronization:
- Manages shared state for portfolio, positions, orders, quotes, etc.
- Handles WebSocket and SSE connections for real-time updates
- Coordinates React Query cache updates
- Provides methods for manual refresh

### 2. Synced Data Hooks (`/src/hooks/useSyncedData.ts`)
Convenience hooks for components to access synchronized data:
- `useSyncedPortfolio()` - Portfolio summary data
- `useSyncedAccount()` - Account balances
- `useSyncedPositions()` - Open positions
- `useSyncedQuotes()` - Real-time quotes
- `useSyncedOrders()` - Open orders
- `useSyncedTrades()` - Recent trades
- `useSyncedDecisions()` - Trading decisions
- `useSyncedStrategies()` - Active strategies
- `useSyncStatus()` - Connection and sync status
- `useRefreshData()` - Manual refresh triggers

### 3. Data Flow

```
Tradier API
    ↓
Backend (minimal_server.js)
    ↓
REST API + WebSocket + SSE
    ↓
DataSyncContext
    ↓
React Query Cache
    ↓
Component Hooks
    ↓
UI Components
```

## How It Works

### Initial Load
1. When the app starts, `DataSyncContext` fetches all data in parallel
2. Data is stored in both local state and React Query cache
3. Components using synced hooks get immediate access to cached data

### Real-time Updates
1. **WebSocket**: Handles portfolio, position, quote, and strategy updates
2. **SSE**: Handles order status updates via `/api/paper/orders/stream`
3. Updates flow through `DataSyncContext` which:
   - Updates local state
   - Updates React Query cache
   - Triggers component re-renders

### Periodic Refresh
- Configurable intervals in `/src/config/dataRefreshConfig.ts`
- Different data types refresh at appropriate intervals:
  - Quotes: 3 seconds
  - Orders/Trades: 3-5 seconds
  - Portfolio/Positions: 10 seconds
  - Strategies/Context: 30 seconds

## Usage Example

```tsx
import { useSyncedPortfolio, useSyncedPositions } from '@/hooks/useSyncedData';

function MyComponent() {
  const { data: portfolio, isLoading: portfolioLoading } = useSyncedPortfolio();
  const { data: positions, isLoading: positionsLoading } = useSyncedPositions();
  
  if (portfolioLoading || positionsLoading) {
    return <div>Loading...</div>;
  }
  
  return (
    <div>
      <h2>Portfolio Value: ${portfolio?.equity}</h2>
      <ul>
        {positions?.map(pos => (
          <li key={pos.symbol}>{pos.symbol}: {pos.qty} shares</li>
        ))}
      </ul>
    </div>
  );
}
```

## Benefits

1. **Consistency**: All components see the same data
2. **Performance**: Data is fetched once and shared
3. **Real-time**: WebSocket/SSE updates propagate instantly
4. **Resilience**: Automatic reconnection and error recovery
5. **Developer Experience**: Simple hooks API

## Monitoring

The `DataSyncDashboard` component shows:
- Connection status
- Last sync time for each data source
- Number of items in each category
- Overall sync health

Use the `SyncStatusIndicator` component in headers/toolbars to show connection status.

## Configuration

Edit `/src/config/dataRefreshConfig.ts` to adjust:
- Refresh intervals
- Stale time thresholds
- WebSocket reconnection settings
- SSE retry configuration
