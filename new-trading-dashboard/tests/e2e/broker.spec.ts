import { test, expect } from '@playwright/test';

test.describe('Broker Intelligence Page', () => {
  test.beforeEach(async ({ page }) => {
    // Mock backend responses for testing
    await page.route('**/api/broker/performance', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          'Primary Broker (Tradier)': {
            pnl: 1250.75,
            fillRate: 0.98,
            slippage: 0.12,
            latency: 42,
            errorRate: 0.02,
            uptime: 0.999,
            status: 'online',
            isActive: true
          },
          'Secondary Broker (Alpaca)': {
            pnl: 950.25,
            fillRate: 0.96,
            slippage: 0.18,
            latency: 48,
            errorRate: 0.04,
            uptime: 0.995,
            status: 'online',
            isActive: false
          }
        })
      });
    });

    await page.route('**/api/alerts', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 'alert-1',
            type: 'DRAWDOWN',
            message: 'Portfolio drawdown exceeded 5% threshold',
            severity: 'HIGH',
            timestamp: new Date().toISOString(),
            source: 'risk_monitor',
            details: { threshold: 0.05, current: 0.062 },
            acknowledged: false
          },
          {
            id: 'alert-2',
            type: 'BROKER_ISSUE',
            message: 'Latency spike detected on primary broker',
            severity: 'MEDIUM',
            timestamp: new Date(Date.now() - 300000).toISOString(),
            source: 'broker_monitor',
            details: { latency: 180 },
            acknowledged: false
          }
        ])
      });
    });

    await page.route('**/api/strategies/active', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 'strategy-1',
            name: 'VWAP Momentum',
            description: 'Follows momentum using VWAP as reference',
            category: 'momentum',
            enabled: true,
            performance: {
              daily: 1.25,
              weekly: -0.42,
              monthly: 3.17,
              overall: 12.53
            },
            marketSuitability: 0.78,
            parameters: { lookback: 20, threshold: 0.02 },
            status: 'active',
            lastSignal: {
              direction: 'long',
              strength: 0.85,
              timestamp: new Date(Date.now() - 1200000).toISOString()
            }
          },
          {
            id: 'strategy-2',
            name: 'Mean Reversion',
            description: 'Trades mean reversion on oversold conditions',
            category: 'reversion',
            enabled: false,
            performance: {
              daily: -0.35,
              weekly: 1.22,
              monthly: 2.45,
              overall: 8.73
            },
            marketSuitability: 0.45,
            parameters: { bands: 2.5, period: 14 },
            status: 'idle',
            lastSignal: {
              direction: 'neutral',
              strength: 0.2,
              timestamp: new Date(Date.now() - 3600000).toISOString()
            }
          }
        ])
      });
    });

    // Mock failover API call
    await page.route('**/api/broker/failover', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ 
          success: true, 
          activeBroker: 'Secondary Broker (Alpaca)' 
        })
      });
    });

    // Mock strategy toggle API call
    await page.route('**/api/strategies/**/toggle', (route) => {
      const strategyId = route.request().url().split('/')[5];
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ 
          success: true, 
          strategy: {
            id: strategyId,
            enabled: true,
            // Include other fields as needed
          }
        })
      });
    });

    // Navigate to broker page
    await page.goto('/broker');
  });

  test('should display broker performance cards', async ({ page }) => {
    // Check if primary broker card is displayed
    await expect(page.getByText('Primary Broker (Tradier)')).toBeVisible();
    await expect(page.getByText('$1250.75')).toBeVisible();
    await expect(page.getByText('98.0%')).toBeVisible(); // Fill rate

    // Check if secondary broker card is displayed
    await expect(page.getByText('Secondary Broker (Alpaca)')).toBeVisible();
    await expect(page.getByText('$950.25')).toBeVisible();
    await expect(page.getByText('96.0%')).toBeVisible(); // Fill rate
  });

  test('should perform broker failover successfully', async ({ page }) => {
    // Click failover button on secondary broker
    await page.getByText('Failover').click();
    
    // Verify toast message appears
    await expect(page.getByText('Successfully failed over to Secondary Broker (Alpaca)')).toBeVisible();
  });

  test('should display risk alerts properly', async ({ page }) => {
    // Check if alerts are displayed
    await expect(page.getByText('Portfolio drawdown exceeded 5% threshold')).toBeVisible();
    await expect(page.getByText('Latency spike detected on primary broker')).toBeVisible();
    
    // Filter by severity
    await page.getByLabel('Filter alerts by severity').selectOption('HIGH');
    await expect(page.getByText('Portfolio drawdown exceeded 5% threshold')).toBeVisible();
    await expect(page.getByText('Latency spike detected on primary broker')).not.toBeVisible();
    
    // Acknowledge an alert
    await page.getByText('Acknowledge').first().click();
    
    // Toggle to show acknowledged alerts
    await page.getByTitle('Show acknowledged alerts').click();
    await expect(page.getByText('Portfolio drawdown exceeded 5% threshold')).toBeVisible();
  });

  test('should display strategy monitor and handle toggles', async ({ page }) => {
    // Check if strategies are displayed
    await expect(page.getByText('VWAP Momentum')).toBeVisible();
    await expect(page.getByText('Mean Reversion')).toBeVisible();
    
    // Expand a strategy
    await page.locator('button').filter({ has: page.getByText('VWAP Momentum') }).locator('svg').last().click();
    await expect(page.getByText('Daily')).toBeVisible();
    await expect(page.getByText('+1.25%')).toBeVisible();
    
    // Toggle a strategy
    const switchElement = page.locator('input[type="checkbox"]').nth(1);
    await switchElement.check();
    
    // Verify toast message
    await expect(page.getByText('Strategy enabled successfully')).toBeVisible();
  });
});
