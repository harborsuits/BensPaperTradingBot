import React, { useState } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import WebSocketIndicator from '@/components/ui/WebSocketIndicator';
import HealthPill from '@/components/HealthPill';
import HealthBadge from '@/components/ui/HealthBadge';
import { useHealth } from '@/hooks/useHealth';

// Import icons
import { 
  LayoutDashboard, 
  BarChart3, 
  LineChart, 
  Settings, 
  Briefcase, 
  Brain, 
  FileText, 
  Bell,
  LogOut,
  Menu,
  X,
  ChevronDown,
  ChevronUp,
  TrendingUp
} from 'lucide-react';

const MainLayout: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isAccountMenuOpen, setIsAccountMenuOpen] = useState(false);
  const { data: health } = useHealth();
  const offline = !health || (health as any)?.ok === false || (health as any)?.status === 'down';
  
  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/portfolio', label: 'Portfolio', icon: <Briefcase size={20} /> },
    { path: '/strategies', label: 'Strategies', icon: <Settings size={20} /> },
    { path: '/decisions', label: 'Trade Decisions', icon: <BarChart3 size={20} /> },
    { path: '/context', label: 'Market Context', icon: <LineChart size={20} /> },
    { path: '/market', label: 'Market Data', icon: <TrendingUp size={20} /> },
    { path: '/evotester', label: 'EvoTester', icon: <Brain size={20} /> },
    { path: '/logs', label: 'Logs & Alerts', icon: <FileText size={20} /> },
  ];

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Sidebar */}
      <aside 
        className={`${
          isSidebarOpen ? 'w-64' : 'w-20'
        } bg-card border-r border-border hidden md:block transition-all duration-300 ease-in-out`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 flex items-center justify-between">
            <h1 className={`font-bold text-xl ${isSidebarOpen ? 'block' : 'hidden'}`}>BenBot Trading</h1>
            <button 
              onClick={toggleSidebar} 
              className="p-2 rounded-md hover:bg-muted"
            >
              <Menu size={20} />
            </button>
          </div>
          
          <nav className="flex-1 py-4">
            <ul className="space-y-1 px-2 list-none">
              {navItems.map((item) => (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) => `
                      flex items-center gap-3 px-3 py-2 rounded-md transition-colors
                      ${isActive ? 'bg-primary/10 text-primary' : 'hover:bg-muted text-foreground'}
                    `}
                  >
                    {item.icon}
                    {isSidebarOpen && <span>{item.label}</span>}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          
          <div className="p-4 border-t border-border">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
                {user?.username && user.username[0]?.toUpperCase()}
              </div>
              {isSidebarOpen && (
                <div>
                  <p className="font-medium">{user?.username}</p>
                  <button 
                    onClick={handleLogout}
                    className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
                  >
                    <LogOut size={14} />
                    Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </aside>
      
      {/* Mobile sidebar overlay */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}
      
      {/* Mobile sidebar */}
      <div 
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border transform ${
          isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        } transition-transform duration-300 ease-in-out md:hidden`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 flex items-center justify-between">
            <h1 className="font-bold text-xl">BenBot Trading</h1>
            <button 
              onClick={() => setIsMobileMenuOpen(false)} 
              className="p-2 rounded-md hover:bg-muted"
            >
              <X size={20} />
            </button>
          </div>
          
          <nav className="flex-1 py-4">
            <ul className="space-y-1 px-2 list-none">
              {navItems.map((item) => (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) => `
                      flex items-center gap-3 px-3 py-2 rounded-md transition-colors
                      ${isActive ? 'bg-primary/10 text-primary' : 'hover:bg-muted text-foreground'}
                    `}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          
          <div className="p-4 border-t border-border">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
                {user?.username && user.username[0]?.toUpperCase()}
              </div>
              <div>
                <p className="font-medium">{user?.username}</p>
                <button 
                  onClick={handleLogout}
                  className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
                >
                  <LogOut size={14} />
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
              {/* Main content */}
      <div className="flex flex-col flex-1 min-h-screen">
        {/* Top bar */}
        <header className="h-16 border-b border-border bg-card flex items-center justify-between px-4">
          <div className="flex items-center">
            <button 
              onClick={() => setIsMobileMenuOpen(true)} 
              className="p-2 rounded-md hover:bg-muted md:hidden"
            >
              <Menu size={20} />
            </button>
          </div>
          
          <div className="flex items-center gap-4">
            <WebSocketIndicator className="mr-2" showText={false} size={20} />
            {/* Existing HealthPill retained; add breaker-aware badge */}
            <HealthPill />
            {health && (
              <HealthBadge state={(health as any).breaker || 'AMBER'} asOf={(health as any)?.meta?.asOf || (health as any)?.asOf} />
            )}
            
            <button className="p-2 rounded-full hover:bg-muted relative">
              <Bell size={20} />
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-primary"></span>
            </button>
            
            <div className="relative">
              <button 
                onClick={() => setIsAccountMenuOpen(!isAccountMenuOpen)}
                className="flex items-center gap-2 p-2 rounded-md hover:bg-muted"
              >
                <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
                  {user?.username && user.username[0]?.toUpperCase()}
                </div>
                <span className="hidden sm:block">{user?.username}</span>
                {isAccountMenuOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              
              {isAccountMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 py-2 bg-card border border-border rounded-md shadow-lg z-10">
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-4 py-2 hover:bg-muted flex items-center gap-2"
                  >
                    <LogOut size={16} />
                    Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        {offline && (
          <div className="bg-yellow-100 text-yellow-900 text-sm px-4 py-2 border-b border-yellow-300">
            Engine offline — showing last-known data. Some tiles may pause until the API recovers.
          </div>
        )}
        
        {/* Content area with AppLayout wrapper */}
        <main className="flex-1 overflow-auto">
          <div className="px-4">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
