export const qk = {
  portfolio: (mode: 'paper' | 'live') => ['portfolio', mode] as const,
  decisions: (limit = 50) => ['decisions', 'recent', limit] as const,
  logs: (level: 'ALL' | 'INFO' | 'WARNING' | 'ERROR' = 'INFO') => ['logs', level] as const,
  strategies: ['strategies', 'all'] as const,
  context: ['context'] as const,
  safety: ['safety'] as const,
};


