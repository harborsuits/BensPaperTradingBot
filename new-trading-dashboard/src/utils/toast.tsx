import toast from 'react-hot-toast';

interface ToastOptions {
  duration?: number;
  id?: string;
}

const defaultOptions: ToastOptions = {
  duration: 5000,
};

export const showSuccessToast = (message: string, options: ToastOptions = {}) => {
  return toast.success(message, {
    ...defaultOptions,
    ...options,
    className: 'bg-green-50 text-green-800 border-l-4 border-green-500',
    icon: '✅',
  });
};

export const showErrorToast = (message: string, options: ToastOptions = {}) => {
  return toast.error(message, {
    ...defaultOptions,
    ...options,
    className: 'bg-red-50 text-red-800 border-l-4 border-red-500',
    icon: '❌',
  });
};

export const showInfoToast = (message: string, options: ToastOptions = {}) => {
  return toast(message, {
    ...defaultOptions,
    ...options,
    className: 'bg-blue-50 text-blue-800 border-l-4 border-blue-500',
    icon: 'ℹ️',
  });
};

export const showLoadingToast = (message: string, options: ToastOptions = {}) => {
  return toast.loading(message, {
    ...defaultOptions,
    ...options,
    className: 'bg-gray-50 text-gray-800 border-l-4 border-gray-500',
  });
};

export const dismissToast = (id: string) => {
  toast.dismiss(id);
};

export const updateToast = (id: string, message: string, type: 'success' | 'error' | 'loading' | 'info' = 'info') => {
  toast.dismiss(id);
  
  switch (type) {
    case 'success':
      showSuccessToast(message, { id });
      break;
    case 'error':
      showErrorToast(message, { id });
      break;
    case 'loading':
      showLoadingToast(message, { id });
      break;
    default:
      showInfoToast(message, { id });
  }
};

// For more complex notifications with actions
export const showActionToast = (message: string, actionText: string, onAction: () => void, options: ToastOptions = {}) => {
  return toast.custom(
    (t) => (
      <div
        className={`${
          t.visible ? 'animate-enter' : 'animate-leave'
        } max-w-md w-full bg-white shadow-lg rounded-lg pointer-events-auto flex ring-1 ring-black ring-opacity-5`}
      >
        <div className="flex-1 w-0 p-4">
          <div className="flex items-start">
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium text-gray-900">{message}</p>
            </div>
          </div>
        </div>
        <div className="flex border-l border-gray-200">
          <button
            onClick={() => {
              onAction();
              toast.dismiss(t.id);
            }}
            className="w-full border border-transparent rounded-none rounded-r-lg p-4 flex items-center justify-center text-sm font-medium text-blue-600 hover:text-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {actionText}
          </button>
        </div>
      </div>
    ),
    options
  );
};
