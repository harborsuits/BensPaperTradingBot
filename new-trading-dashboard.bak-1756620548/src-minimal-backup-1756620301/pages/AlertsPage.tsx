import React from 'react';
import { Helmet } from 'react-helmet-async';
import { Bell } from 'lucide-react';

import AlertsFeed from '@/components/alerts/AlertsFeed';
import { ToastProvider } from '@/components/ui/Toast';

/**
 * AlertsPage serves as a dedicated page for viewing and managing all alerts and notifications
 */
const AlertsPage: React.FC = () => {
  return (
    <>
      <Helmet>
        <title>Alerts & Notifications | BenBot Trading Dashboard</title>
      </Helmet>
      
      <ToastProvider>
        <div className="container py-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Bell className="text-primary" size={24} />
              Alerts & Notifications
            </h1>
          </div>

          <div className="space-y-6">
            <AlertsFeed />
          </div>
        </div>
      </ToastProvider>
    </>
  );
};

export default AlertsPage;
