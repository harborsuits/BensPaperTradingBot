import React, { createContext, useContext, useState, useEffect } from 'react';
import { authApi } from '@/services/api';
import { AuthRequest } from '@/types/api.types';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: { username: string } | null;
  login: (credentials: AuthRequest) => Promise<boolean>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [user, setUser] = useState<{ username: string } | null>(null);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        try {
          // Optional: validate token with backend if needed
          // For now, just consider having a token as being authenticated
          
          // Extract username from token if possible (from JWT payload)
          // This is a simplified example - in a real app, you might want to validate the token
          // with the server or decode it properly
          const username = localStorage.getItem('username') || 'Trader';
          setUser({ username });
          setIsAuthenticated(true);
        } catch (error) {
          localStorage.removeItem('auth_token');
          localStorage.removeItem('username');
        }
      }
      
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const login = async (credentials: AuthRequest): Promise<boolean> => {
    setIsLoading(true);
    try {
      const response = await authApi.login(credentials);
      
      if (response.success && response.data) {
        localStorage.setItem('auth_token', response.data.access_token);
        localStorage.setItem('username', credentials.username);
        setUser({ username: credentials.username });
        setIsAuthenticated(true);
        return true;
      }
      
      return false;
    } catch (error) {
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async (): Promise<void> => {
    setIsLoading(true);
    try {
      await authApi.logout();
    } finally {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('username');
      setUser(null);
      setIsAuthenticated(false);
      setIsLoading(false);
    }
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
