# BensBot UI Design Specification

## Overview
This document outlines the visual and interaction design for BensBot's enhanced dashboard UI, inspired by elements from trading-dashboard, MyCryptoBot, and Crypto-Bot repositories, but adapted specifically for BensBot's architecture and data model.

## Design Principles
- **Responsive & Clean**: Minimal, data-focused UI that works on all devices
- **Real-time Focused**: Emphasis on live data and status
- **Autonomous Trading-Centric**: UI tailored to monitor autonomous trading performance
- **MongoDB Integration**: Direct data retrieval from BensBot's existing MongoDB

## Core UI Sections

### 1. Navigation & Layout
- **Top Bar**: Navigation + account type selector (Paper/Live) + time frame selectors
- **Side Panel**: Bot controls, filter options, and system status
- **Main Content**: Tab-based content with portfolio, positions, orders, and analytics
- **Color Theme**: Professional trading platform palette (dark mode optional)

### 2. Portfolio Dashboard
- **KPI Cards**: Inspired by trading-dashboard's clean metrics display
  - Total Equity
  - Daily P&L (with % change)
  - Open Positions Count
  - Winning Trades Ratio
  
- **Equity Chart**: Interactive time-series visualization (from MyCryptoBot concept)
  - Time period selectors (1D, 1W, 1M, YTD, ALL)
  - Comparison with S&P 500 (overlay option)
  - Annotated trade markers

- **Allocation Chart**: Portfolio breakdown pie/treemap
  - Symbol distribution
  - Sector allocation
  - Risk exposure visualization

### 3. Positions Monitor
- **Active Positions Table**: Clean, sortable table with:
  - Symbol with mini-chart sparkline
  - Entry price, current price, P&L
  - Position size and allocation percentage
  - Duration and unrealized gain/loss
  - Risk metrics (stop distance, etc.)

- **Watch List Component**: Inspired by trading-dashboard
  - Real-time quotes with color-coded price changes
  - Quick view technical indicators
  - Add/remove symbols functionality

### 4. Order History & Analytics
- **Orders Table**: Filterable transaction history
  - Execution timeline
  - Status indicators
  - Trade type and parameters
  
- **Performance Analytics**: Adapted from MyCryptoBot
  - Win/loss ratio
  - Average gain per trade
  - Drawdown chart
  - Trading strategy performance breakdown

### 5. Bot Control Interface
- **Bot Status Card**: Clear visual status of the trading bot
  - On/Off state with visual indicator
  - Last action timestamp
  - Current status message

- **Control Panel**: Inspired by Crypto-Bot
  - Start/Stop buttons for paper trading
  - Risk parameter adjustments
  - Market status indicators
  - Recent activity log

## Technical Implementation Notes

### Streamlit Enhancements
- Custom CSS to match professional trading platforms
- JavaScript components for real-time updates
- Cached data retrieval patterns for MongoDB

### Data Integration
- Direct connection to BensBot's MongoDB for:
  - Account data
  - Position information
  - Order history
  - Performance metrics

### Real-time Features
- Auto-refresh components
- WebSocket connection for price updates
- Trade notification alerts

## Mobile Considerations
- Responsive layout adjustments
- Touch-friendly controls
- Simplified views for small screens

## Implementation Phases
1. **Core Layout & Theme**: Basic dashboard structure and styling
2. **Portfolio Visualization**: Equity charts and KPI cards
3. **Position Monitoring**: Active positions and watchlist
4. **Order History**: Trade logs and analytics
5. **Bot Controls**: System status and control interface

## Reference Images
*[Include screenshots from inspiration UIs with annotations showing which elements to adapt]*
