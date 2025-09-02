import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './lib/queryClient'
import RQDevtoolsToggle from './shared/RQDevtoolsToggle'
import { Provider } from 'react-redux'
import { store } from './redux/store'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <Provider store={store}>
        <App />
        {import.meta.env.VITE_RQ_DEVTOOLS !== 'false' && <RQDevtoolsToggle />}
      </Provider>
    </QueryClientProvider>
  </React.StrictMode>,
)