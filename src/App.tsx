import { useState } from 'react'
import './App.css'

// Layout components
import MainContent from './components/MainContent'
import { AICoPilot } from './components/AICoPilot'

function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')

  return (
    <div className={`app ${theme} text-white`}>
      <div className="h-screen overflow-hidden bg-background">
        {/* Main Content */}
        <MainContent />
        
        {/* AI Co-Pilot */}
        <AICoPilot />
      </div>
    </div>
  )
}

export default App 