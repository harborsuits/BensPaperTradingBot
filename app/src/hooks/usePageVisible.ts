import { useEffect, useState } from "react";

/**
 * Hook to track whether the page is currently visible
 * Pauses data fetching when the tab is in the background
 * 
 * @returns boolean - true if page is visible, false if hidden
 */
export function usePageVisible() {
  const [isVisible, setIsVisible] = useState(!document.hidden);
  
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsVisible(!document.hidden);
    };
    
    document.addEventListener("visibilitychange", handleVisibilityChange);
    
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);
  
  return isVisible;
}
