import React, { createContext, useContext, useMemo, useState, useEffect } from 'react';

type AuthContextType = {
  isAuthenticated: boolean;
  isLoading: boolean;
  getAccessToken: () => string | null;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    setIsLoading(false);
  }, []);

  const getAccessToken = () => localStorage.getItem('auth_token');

  const value = useMemo<AuthContextType>(
    () => ({ isAuthenticated, isLoading, getAccessToken }),
    [isAuthenticated, isLoading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};


