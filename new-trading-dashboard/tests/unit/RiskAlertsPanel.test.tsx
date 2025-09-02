import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RiskAlertsPanel from '../../src/components/broker/RiskAlertsPanel';
import * as riskAlerts from '../../src/services/riskAlerts';

// Define types for TypeScript
type RiskAlertSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

interface RiskAlert {
  id: string;
  type: string;
  message: string;
  severity: RiskAlertSeverity;
  timestamp: string;
  source: string;
  details: Record<string, any>;
  acknowledged: boolean;
}

// Mock the risk alerts module
jest.mock('../../src/services/riskAlerts', () => ({
  getAlerts: jest.fn(),
  acknowledgeAlert: jest.fn(),
  clearAcknowledgedAlerts: jest.fn(),
  subscribeToAlerts: jest.fn()
}));

describe('RiskAlertsPanel Component', () => {
  const mockAlerts = [
    {
      id: 'alert-1',
      type: 'DRAWDOWN',
      message: 'Portfolio drawdown exceeded 5% threshold',
      severity: 'HIGH' as riskAlerts.RiskAlertSeverity,
      timestamp: new Date().toISOString(),
      source: 'risk_monitor',
      details: { threshold: 0.05, current: 0.062 },
      acknowledged: false
    },
    {
      id: 'alert-2',
      type: 'BROKER_ISSUE',
      message: 'Latency spike detected on primary broker',
      severity: 'MEDIUM' as riskAlerts.RiskAlertSeverity,
      timestamp: new Date(Date.now() - 300000).toISOString(),
      source: 'broker_monitor',
      details: { latency: 180 },
      acknowledged: false
    },
    {
      id: 'alert-3',
      type: 'MARKET_EVENT',
      message: 'Significant price movement in SPY',
      severity: 'LOW' as riskAlerts.RiskAlertSeverity,
      timestamp: new Date(Date.now() - 600000).toISOString(),
      source: 'market_monitor',
      details: { symbol: 'SPY', change: -1.2 },
      acknowledged: true
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock the risk alerts functions
    (riskAlerts.getAlerts as jest.Mock).mockReturnValue(mockAlerts);
    (riskAlerts.subscribeToAlerts as jest.Mock).mockImplementation((callback) => {
      callback(mockAlerts);
      return jest.fn(); // Return unsubscribe function
    });
    (riskAlerts.acknowledgeAlert as jest.Mock).mockImplementation((alertId) => {
      return mockAlerts.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      );
    });
    (riskAlerts.clearAcknowledgedAlerts as jest.Mock).mockImplementation(() => {
      return mockAlerts.filter(alert => !alert.acknowledged);
    });
  });

  test('renders alerts correctly', () => {
    render(<RiskAlertsPanel />);
    
    // Check if alerts are rendered
    expect(screen.getByText('Portfolio drawdown exceeded 5% threshold')).toBeInTheDocument();
    expect(screen.getByText('Latency spike detected on primary broker')).toBeInTheDocument();
    
    // Acknowledged alerts should not be visible by default
    expect(screen.queryByText('Significant price movement in SPY')).not.toBeInTheDocument();
  });

  test('filters alerts by severity', () => {
    render(<RiskAlertsPanel />);
    
    // Get the severity filter dropdown
    const filterDropdown = screen.getByLabelText('Filter alerts by severity');
    
    // Filter by HIGH severity
    fireEvent.change(filterDropdown, { target: { value: 'HIGH' } });
    
    // HIGH alert should be visible
    expect(screen.getByText('Portfolio drawdown exceeded 5% threshold')).toBeInTheDocument();
    
    // MEDIUM alert should not be visible
    expect(screen.queryByText('Latency spike detected on primary broker')).not.toBeInTheDocument();
    
    // Filter by MEDIUM severity
    fireEvent.change(filterDropdown, { target: { value: 'MEDIUM' } });
    
    // MEDIUM alert should be visible
    expect(screen.getByText('Latency spike detected on primary broker')).toBeInTheDocument();
    
    // HIGH alert should not be visible
    expect(screen.queryByText('Portfolio drawdown exceeded 5% threshold')).not.toBeInTheDocument();
  });

  test('handles showing acknowledged alerts', () => {
    render(<RiskAlertsPanel />);
    
    // Acknowledged alerts should not be visible by default
    expect(screen.queryByText('Significant price movement in SPY')).not.toBeInTheDocument();
    
    // Find show acknowledged button by title and click it
    const showAcknowledgedButton = screen.getByTitle('Show acknowledged alerts');
    fireEvent.click(showAcknowledgedButton);
    
    // Now the acknowledged alert should be visible
    expect(screen.getByText('Significant price movement in SPY')).toBeInTheDocument();
  });

  test('acknowledges alerts correctly', () => {
    render(<RiskAlertsPanel />);
    
    // Find and click the acknowledge button for the first alert
    const acknowledgeButtons = screen.getAllByText('Acknowledge');
    fireEvent.click(acknowledgeButtons[0]);
    
    // Check if acknowledgeAlert was called with correct ID
    expect(riskAlerts.acknowledgeAlert).toHaveBeenCalledWith('alert-1');
  });

  test('clears acknowledged alerts', () => {
    render(<RiskAlertsPanel />);
    
    // Find clear acknowledged button by title and click it
    const clearAcknowledgedButton = screen.getByTitle('Clear acknowledged alerts');
    fireEvent.click(clearAcknowledgedButton);
    
    // Check if clearAcknowledgedAlerts was called
    expect(riskAlerts.clearAcknowledgedAlerts).toHaveBeenCalled();
  });

  test('displays empty state when no alerts', () => {
    // Mock empty alerts
    (riskAlerts.getAlerts as jest.Mock).mockReturnValue([]);
    (riskAlerts.subscribeToAlerts as jest.Mock).mockImplementation((callback) => {
      callback([]);
      return jest.fn();
    });
    
    render(<RiskAlertsPanel />);
    
    // Check if empty state message is displayed
    expect(screen.getByText('No alerts')).toBeInTheDocument();
  });
});
