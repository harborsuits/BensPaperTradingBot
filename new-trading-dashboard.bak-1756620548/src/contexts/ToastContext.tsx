import React, { createContext, useState, useContext, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Toast as ToastType, ToastType as NotificationType } from '../components/ui/Toast';
import ToastContainer from '../components/ui/ToastContainer';

interface ToastContextType {
  showToast: (type: NotificationType, message: string, duration?: number) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const useToast = (): ToastContextType => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastType[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prevToasts) => prevToasts.filter((toast) => toast.id !== id));
  }, []);

  const showToast = useCallback(
    (type: NotificationType, message: string, duration = 5000) => {
      const id = uuidv4();
      const newToast: ToastType = {
        id,
        type,
        message,
        duration,
      };
      setToasts((prevToasts) => [...prevToasts, newToast]);
    },
    []
  );

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      <ToastContainer toasts={toasts} onClose={removeToast} />
    </ToastContext.Provider>
  );
};

export default ToastContext;
