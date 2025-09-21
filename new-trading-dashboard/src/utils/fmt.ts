/**
 * Formatting utilities to replace scattered toFixed() calls
 * Provides consistent number and currency formatting
 */

export const fmt = {
  /**
   * Format currency values
   */
  money: (value?: number | null): string => {
    if (value == null || isNaN(value)) return '$0.00';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  },

  /**
   * Format percentages
   */
  pct: (value?: number | null, decimals: number = 2): string => {
    if (value == null || isNaN(value)) return '0.00%';
    return `${(value * 100).toFixed(decimals)}%`;
  },

  /**
   * Format numbers with optional decimal places
   */
  num: (value?: number | null, decimals?: number): string => {
    if (value == null || isNaN(value)) return '0';
    if (decimals !== undefined) {
      return value.toFixed(decimals);
    }
    return new Intl.NumberFormat('en-US').format(value);
  },

  /**
   * Format large numbers with K/M/B suffixes
   */
  bigNum: (value?: number | null): string => {
    if (value == null || isNaN(value)) return '0';

    const absValue = Math.abs(value);
    if (absValue >= 1e9) {
      return `${(value / 1e9).toFixed(1)}B`;
    } else if (absValue >= 1e6) {
      return `${(value / 1e6).toFixed(1)}M`;
    } else if (absValue >= 1e3) {
      return `${(value / 1e3).toFixed(1)}K`;
    }

    return fmt.num(value);
  },

  /**
   * Format dates consistently
   */
  date: (date?: string | Date | null): string => {
    if (!date) return '';
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  },

  /**
   * Format relative time (e.g., "2 hours ago")
   */
  relativeTime: (date?: string | Date | null): string => {
    if (!date) return '';
    const d = typeof date === 'string' ? new Date(date) : date;
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();

    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return fmt.date(d);
  }
};

// Legacy aliases for backward compatibility
export const formatMoney = fmt.money;
export const formatPercent = fmt.pct;
export const formatNumber = fmt.num;
