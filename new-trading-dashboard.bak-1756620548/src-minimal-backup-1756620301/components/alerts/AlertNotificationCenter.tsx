import React, { useState, useEffect } from 'react';
import { loggingApi } from '@/services/api';
import { AlertNotification } from '@/types/api.types';
import { useAlertUpdates } from '@/hooks/useWebSocketSubscriptions';
import { FiBell, FiAlertTriangle, FiInfo, FiX, FiCheck } from 'react-icons/fi';

interface AlertNotificationCenterProps {
  maxAlerts?: number;
}

const AlertNotificationCenter: React.FC<AlertNotificationCenterProps> = ({
  maxAlerts = 5
}) => {
  const [alerts, setAlerts] = useState<AlertNotification[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  // Fetch initial alerts
  useEffect(() => {
    const fetchAlerts = async () => {
      setLoading(true);
      try {
        const response = await loggingApi.getAlerts(undefined, maxAlerts, false);
        if (response.success && response.data) {
          setAlerts(response.data);
          setUnreadCount(response.data.filter(a => !a.acknowledged).length);
        } else {
          setError(response.error || 'Failed to fetch alerts');
        }
      } catch (err) {
        setError('Error fetching alerts');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
  }, [maxAlerts]);

  // Listen for WebSocket alert updates
  const { isConnected } = useAlertUpdates(
    // New alert handler
    (newAlert) => {
      setAlerts(prevAlerts => {
        // Only add if it's not already in the list
        if (!prevAlerts.some(a => a.id === newAlert.id)) {
          const updatedAlerts = [newAlert, ...prevAlerts].slice(0, maxAlerts);
          setUnreadCount(prev => prev + 1);
          return updatedAlerts;
        }
        return prevAlerts;
      });
    },
    // Alert update handler (e.g., when acknowledged)
    (updatedAlert) => {
      setAlerts(prevAlerts => {
        return prevAlerts.map(alert => {
          if (alert.id === updatedAlert.id) {
            // If this was unacknowledged before and is now acknowledged
            if (!alert.acknowledged && updatedAlert.acknowledged) {
              setUnreadCount(prev => Math.max(0, prev - 1));
            }
            return updatedAlert;
          }
          return alert;
        });
      });
    }
  );

  // Acknowledge an alert
  const handleAcknowledge = async (alert: AlertNotification) => {
    try {
      await loggingApi.acknowledgeAlert(alert.id);
      // No need to update the state here as we'll get the WebSocket update
    } catch (err) {
      console.error('Error acknowledging alert:', err);
    }
  };

  // Clear all alerts
  const handleClearAll = () => {
    // In a real implementation, this would call a backend API to mark all as acknowledged
    alerts.forEach(alert => {
      if (!alert.acknowledged) {
        handleAcknowledge(alert);
      }
    });
  };

  // Get icon and color based on severity
  const getSeverityDetails = (severity: string) => {
    switch (severity) {
      case 'critical':
        return {
          icon: <FiAlertTriangle className="h-5 w-5 text-red-500" />,
          bgColor: 'bg-red-900 bg-opacity-50',
          borderColor: 'border-red-700'
        };
      case 'warning':
        return {
          icon: <FiAlertTriangle className="h-5 w-5 text-yellow-500" />,
          bgColor: 'bg-yellow-900 bg-opacity-50',
          borderColor: 'border-yellow-700'
        };
      default: // info
        return {
          icon: <FiInfo className="h-5 w-5 text-blue-500" />,
          bgColor: 'bg-blue-900 bg-opacity-50',
          borderColor: 'border-blue-700'
        };
    }
  };

  return (
    <div className="relative">
      {/* Bell icon with notification indicator */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 rounded-full hover:bg-navy-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <FiBell className="h-6 w-6 text-gray-300" />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 rounded-full bg-red-600">
            {unreadCount}
          </span>
        )}
      </button>

      {/* Alerts dropdown */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 max-h-96 overflow-y-auto rounded-md shadow-lg z-50 bg-navy-800 border border-navy-600">
          <div className="p-3 border-b border-navy-600 flex justify-between items-center">
            <h3 className="text-sm font-medium text-white">Notifications</h3>
            <div className="flex space-x-2">
              {isConnected ? (
                <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
                  Live
                </span>
              ) : (
                <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">
                  Offline
                </span>
              )}
              {unreadCount > 0 && (
                <button
                  onClick={handleClearAll}
                  className="text-xs text-gray-400 hover:text-white"
                >
                  Clear all
                </button>
              )}
            </div>
          </div>

          {loading ? (
            <div className="p-4 text-center text-gray-400">Loading notifications...</div>
          ) : error ? (
            <div className="p-4 text-center text-red-500">{error}</div>
          ) : alerts.length === 0 ? (
            <div className="p-4 text-center text-gray-400">No notifications</div>
          ) : (
            <div>
              {alerts.map((alert) => {
                const { icon, bgColor, borderColor } = getSeverityDetails(alert.severity);
                return (
                  <div
                    key={alert.id}
                    className={`p-3 border-b ${borderColor} ${bgColor} ${
                      !alert.acknowledged ? 'bg-opacity-20' : 'bg-opacity-10'
                    }`}
                  >
                    <div className="flex">
                      <div className="flex-shrink-0 mr-3">{icon}</div>
                      <div className="flex-1">
                        <div className="flex justify-between">
                          <p className="text-sm font-medium text-white">{alert.title}</p>
                          <p className="text-xs text-gray-400">
                            {new Date(alert.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                        <p className="text-xs text-gray-300 mt-1">{alert.message}</p>
                        <div className="mt-2 flex justify-between items-center">
                          <p className="text-xs text-gray-400">From: {alert.source}</p>
                          {!alert.acknowledged && (
                            <button
                              onClick={() => handleAcknowledge(alert)}
                              className="flex items-center text-xs text-blue-400 hover:text-blue-300"
                            >
                              <FiCheck className="mr-1" />
                              Acknowledge
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AlertNotificationCenter;
