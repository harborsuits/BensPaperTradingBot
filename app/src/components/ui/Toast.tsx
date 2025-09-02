import React, { useEffect, useState, createContext, useContext } from 'react';
import { X, AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react';

export type ToastVariant = 'default' | 'destructive' | 'success' | 'warning';

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant: ToastVariant;
  duration?: number;
  action?: ToastAction;
}

interface ToastProps {
  toast: Toast;
  onClose: () => void;
}

type ToastContextType = {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  removeAllToasts: () => void;
};

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = (toast: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { ...toast, id }]);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  };

  const removeAllToasts = () => {
    setToasts([]);
  };

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, removeAllToasts }}>
      {children}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
        {toasts.map((toast) => (
          <Toast key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

export const createToast = (toast: Omit<Toast, 'id'>) => {
  const context = useContext(ToastContext);
  if (!context) {
    console.error('ToastProvider not found');
    return;
  }
  context.addToast(toast);
};

const Toast: React.FC<ToastProps> = ({ toast, onClose }) => {
  useEffect(() => {
    if (toast.duration !== undefined) {
      const timer = setTimeout(() => {
        onClose();
      }, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast, onClose]);

  const variantStyles = {
    default: 'bg-background border border-border',
    destructive: 'bg-destructive/15 border-l-4 border-l-destructive text-destructive',
    success: 'bg-green-100 border-l-4 border-l-green-500 text-green-800',
    warning: 'bg-amber-100 border-l-4 border-l-amber-500 text-amber-800',
  }[toast.variant];

  const Icon = {
    default: Info,
    destructive: AlertCircle,
    success: CheckCircle,
    warning: AlertTriangle,
  }[toast.variant];

  return (
    <div
      className={`w-full rounded-lg shadow-md p-4 flex ${variantStyles}`}
      role="alert"
    >
      <div className="flex-shrink-0">
        <Icon className="h-5 w-5" />
      </div>
      <div className="ml-3 flex-grow">
        <div className="font-medium">{toast.title}</div>
        {toast.description && (
          <div className="text-sm mt-1 opacity-90">{toast.description}</div>
        )}
        {toast.action && (
          <button 
            onClick={toast.action.onClick}
            className="text-sm mt-2 underline font-medium hover:opacity-80"
          >
            {toast.action.label}
          </button>
        )}
      </div>
      <button
        onClick={onClose}
        className="ml-auto -mx-1.5 -my-1.5 rounded-lg p-1.5 hover:bg-opacity-20 hover:bg-gray-500 focus:ring-2 focus:ring-gray-400"
      >
        <span className="sr-only">Close</span>
        <X className="h-4 w-4" />
      </button>
    </div>
  );
};

export default Toast;

