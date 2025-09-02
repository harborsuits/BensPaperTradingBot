import { useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// Define keyboard shortcut mappings
export const SHORTCUTS = {
  DASHBOARD: 'g d',
  PORTFOLIO: 'g p',
  STRATEGIES: 'g s',
  CONTEXT: 'g c',
  DECISIONS: 'g t',
  LOGS: 'g l',
  TOGGLE_PAUSE: ' ', // Space key
};

/**
 * Hook to handle keyboard shortcuts throughout the app
 */
export function useKeyboardShortcuts(onPauseToggle?: () => void) {
  const navigate = useNavigate();
  
  const keySequence = useCallback((keys: string[]) => {
    // Create a sequential key matcher
    let buffer: string[] = [];
    let timer: number | null = null;
    
    return (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (document.activeElement instanceof HTMLInputElement || 
          document.activeElement instanceof HTMLTextAreaElement ||
          document.activeElement instanceof HTMLSelectElement) {
        return;
      }
      
      // Clear buffer after delay
      if (timer) clearTimeout(timer);
      
      // Add key to buffer
      buffer.push(e.key.toLowerCase());
      
      // Only keep the length we need
      if (buffer.length > keys.length) {
        buffer = buffer.slice(-keys.length);
      }
      
      // Check if buffer matches keys
      const matched = buffer.join('') === keys.join('');
      
      // Reset buffer after delay
      timer = window.setTimeout(() => {
        buffer = [];
      }, 500) as unknown as number;
      
      return matched;
    };
  }, []);
  
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Handle navigation shortcuts (g + key)
    if (keySequence(['g', 'd'])(e)) navigate('/');
    if (keySequence(['g', 'p'])(e)) navigate('/portfolio');
    if (keySequence(['g', 's'])(e)) navigate('/strategies');
    if (keySequence(['g', 'c'])(e)) navigate('/context');
    if (keySequence(['g', 't'])(e)) navigate('/decisions');
    if (keySequence(['g', 'l'])(e)) navigate('/logs');
    
    // Handle space for pause toggle
    if (e.key === ' ' && onPauseToggle && 
        !(document.activeElement instanceof HTMLInputElement || 
          document.activeElement instanceof HTMLTextAreaElement ||
          document.activeElement instanceof HTMLSelectElement)) {
      e.preventDefault(); // Prevent scrolling
      onPauseToggle();
    }
  }, [navigate, onPauseToggle, keySequence]);
  
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}
