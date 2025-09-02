import React, { createContext, useContext, useState, useEffect } from 'react';
import { useLocation, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import AIChatWidget from '@/components/chat/AIChatWidget';
import { aiChatApi, TradingContext } from '@/services/aiChatApi';
import aiContextAdapter from '@/services/aiContextAdapter';

interface AIChatContextType {
  currentContext: TradingContext;
  updateContext: (contextUpdate: Partial<TradingContext>) => void;
}

const AIChatContext = createContext<AIChatContextType | undefined>(undefined);

export const useAIChat = () => {
  const context = useContext(AIChatContext);
  if (context === undefined) {
    throw new Error('useAIChat must be used within an AIChatProvider');
  }
  return context;
};

interface AIChatProviderProps {
  children: React.ReactNode;
}

export const AIChatProvider: React.FC<AIChatProviderProps> = ({ children }) => {
  const [currentContext, setCurrentContext] = useState<TradingContext>({
    currentTab: '',
    assetClasses: [],
  });
  
  const location = useLocation();
  
  // Extract URL parameters (like symbol) if present
  const params = useParams<{ symbol?: string }>();
  const symbol = params.symbol || '';
  
  // Get the current trading context from the API
  const { data: basicContextData } = useQuery(['ai', 'context'], 
    () => aiChatApi.getTradingContext(), {
      refetchOnWindowFocus: false,
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  );
  
  // Get enriched context with full orchestrator data
  const { data: enrichedContext } = useQuery(
    ['ai', 'enriched-context', basicContextData?.data?.currentTab, symbol],
    async () => {
      if (!basicContextData?.data) return undefined;
      try {
        return await aiContextAdapter.enrichTradingContext({
          ...basicContextData.data,
          symbol: symbol || basicContextData.data.symbol
        });
      } catch (error) {
        console.error('Error enriching context:', error);
        return basicContextData.data;
      }
    },
    {
      enabled: !!basicContextData,
      refetchInterval: 60000, // Refresh every minute
      staleTime: 30000,
    }
  );
  
  // Get symbol-specific context data if viewing a symbol
  const { data: symbolContext } = useQuery(
    ['ai', 'symbol-context', symbol],
    () => symbol ? aiContextAdapter.getSymbolContext(symbol) : Promise.resolve(undefined),
    {
      enabled: !!symbol,
      refetchInterval: symbol ? 30000 : false, // Only refresh if we have a symbol
    }
  );
  
  // Update context when location changes
  useEffect(() => {
    // Extract current tab from pathname
    const pathname = location.pathname;
    const currentTab = pathname.split('/')[1] || 'dashboard';
    
    setCurrentContext(prev => ({
      ...prev,
      currentTab: currentTab.charAt(0).toUpperCase() + currentTab.slice(1),
    }));
  }, [location]);
  
  // Update context when API data changes
  useEffect(() => {
    if (enrichedContext) {
      setCurrentContext(prev => ({
        ...prev,
        ...enrichedContext,
        symbolData: symbolContext?.data,
        // Generate a context prompt for better AI interactions
        contextPrompt: aiContextAdapter.createContextPrompt(enrichedContext)
      }));
    } else if (basicContextData?.data) {
      // Fallback to basic context if enriched isn't available
      setCurrentContext(prev => ({
        ...prev,
        ...basicContextData.data,
        symbolData: symbolContext?.data,
      }));
    }
  }, [enrichedContext, basicContextData, symbolContext]);
  
  // Subscribe to real-time data updates
  useEffect(() => {
    // This would connect to the WebSocket for real-time updates
    // to trading context data - implemented in the WebSocketProvider
    
    // Function to handle WebSocket updates to trading context
    const handleContextUpdate = (event: CustomEvent<Partial<TradingContext>>) => {
      if (event.detail) {
        setCurrentContext(prev => ({
          ...prev,
          ...event.detail,
        }));
      }
    };
    
    // Add event listener for context updates
    window.addEventListener('tradingContextUpdate', handleContextUpdate as EventListener);
    
    return () => {
      // Clean up listeners
      window.removeEventListener('tradingContextUpdate', handleContextUpdate as EventListener);
    };
  }, []);
  
  // Method to update context from components
  const updateContext = (contextUpdate: Partial<TradingContext>) => {
    setCurrentContext(prev => ({
      ...prev,
      ...contextUpdate,
    }));
  };
  
  const value = { 
    currentContext,
    updateContext,
  };
  
  return (
    <AIChatContext.Provider value={value}>
      {children}
      <AIChatWidget contextData={currentContext} />
    </AIChatContext.Provider>
  );
};

export default AIChatProvider;
