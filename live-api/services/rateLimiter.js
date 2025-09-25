/**
 * Rate Limiter Service
 * 
 * Manages API call rates to prevent hitting provider limits and incurring overages.
 * Implements token bucket algorithm with per-provider configurations.
 */

class RateLimiter {
  constructor() {
    // Provider-specific rate limits
    this.providers = {
      tradier: {
        limits: {
          quotes: { requests: 60, window: 60000 }, // 60 per minute
          bars: { requests: 120, window: 60000 }, // 120 per minute  
          orders: { requests: 10, window: 60000 }, // 10 per minute
          account: { requests: 30, window: 60000 }, // 30 per minute
          options: { requests: 30, window: 60000 }  // 30 per minute
        },
        buckets: {},
        queue: []
      },
      finnhub: {
        limits: {
          quotes: { requests: 60, window: 60000 }, // 60 per minute
          news: { requests: 30, window: 60000 },   // 30 per minute
          sentiment: { requests: 30, window: 60000 } // 30 per minute
        },
        buckets: {},
        queue: []
      },
      marketaux: {
        limits: {
          news: { requests: 100, window: 86400000 } // 100 per day (free tier)
        },
        buckets: {},
        queue: []
      },
      newsapi: {
        limits: {
          everything: { requests: 100, window: 86400000 } // 100 per day (free tier)
        },
        buckets: {},
        queue: []
      },
      alpaca: {
        limits: {
          quotes: { requests: 200, window: 60000 }, // 200 per minute
          bars: { requests: 200, window: 60000 },   // 200 per minute
          orders: { requests: 200, window: 60000 }  // 200 per minute
        },
        buckets: {},
        queue: []
      }
    };
    
    // Stats tracking
    this.stats = {
      totalRequests: 0,
      blockedRequests: 0,
      queuedRequests: 0,
      providerStats: {}
    };
    
    // Initialize token buckets
    this.initializeBuckets();
    
    // Start queue processor
    this.startQueueProcessor();
    
    console.log('[RateLimiter] Initialized with provider limits');
  }
  
  /**
   * Initialize token buckets for each provider/endpoint
   */
  initializeBuckets() {
    for (const [provider, config] of Object.entries(this.providers)) {
      this.stats.providerStats[provider] = {
        requests: 0,
        blocked: 0,
        queued: 0
      };
      
      for (const [endpoint, limits] of Object.entries(config.limits)) {
        config.buckets[endpoint] = {
          tokens: limits.requests,
          lastRefill: Date.now(),
          maxTokens: limits.requests,
          refillRate: limits.requests / limits.window,
          window: limits.window
        };
      }
    }
  }
  
  /**
   * Check if request can proceed or should be queued/blocked
   */
  async checkLimit(provider, endpoint, priority = 'normal') {
    this.stats.totalRequests++;
    
    const providerConfig = this.providers[provider];
    if (!providerConfig) {
      console.warn(`[RateLimiter] Unknown provider: ${provider}`);
      return { allowed: true, queued: false };
    }
    
    const bucket = providerConfig.buckets[endpoint];
    if (!bucket) {
      console.warn(`[RateLimiter] Unknown endpoint: ${provider}/${endpoint}`);
      return { allowed: true, queued: false };
    }
    
    // Refill tokens based on time elapsed
    this.refillBucket(bucket);
    
    // Check if we have tokens available
    if (bucket.tokens >= 1) {
      bucket.tokens--;
      this.stats.providerStats[provider].requests++;
      return { allowed: true, queued: false };
    }
    
    // No tokens available
    if (priority === 'high') {
      // High priority requests get queued
      return this.queueRequest(provider, endpoint, priority);
    } else {
      // Normal priority requests are blocked
      this.stats.blockedRequests++;
      this.stats.providerStats[provider].blocked++;
      
      const timeToNextToken = Math.ceil((1 - bucket.tokens) / bucket.refillRate);
      
      return {
        allowed: false,
        queued: false,
        retryAfter: timeToNextToken,
        reason: `Rate limit exceeded for ${provider}/${endpoint}. Retry after ${timeToNextToken}ms`
      };
    }
  }
  
  /**
   * Refill tokens based on time elapsed
   */
  refillBucket(bucket) {
    const now = Date.now();
    const timePassed = now - bucket.lastRefill;
    const tokensToAdd = (timePassed * bucket.refillRate) / 1000;
    
    bucket.tokens = Math.min(bucket.maxTokens, bucket.tokens + tokensToAdd);
    bucket.lastRefill = now;
  }
  
  /**
   * Queue a high-priority request
   */
  queueRequest(provider, endpoint, priority) {
    const request = {
      provider,
      endpoint,
      priority,
      timestamp: Date.now(),
      promise: null
    };
    
    // Create a promise that will be resolved when the request can proceed
    const promise = new Promise((resolve) => {
      request.resolve = resolve;
    });
    
    request.promise = promise;
    
    // Add to provider's queue
    this.providers[provider].queue.push(request);
    this.stats.queuedRequests++;
    this.stats.providerStats[provider].queued++;
    
    return {
      allowed: false,
      queued: true,
      promise: promise
    };
  }
  
  /**
   * Process queued requests
   */
  startQueueProcessor() {
    setInterval(() => {
      for (const [provider, config] of Object.entries(this.providers)) {
        if (config.queue.length === 0) continue;
        
        // Sort queue by priority and timestamp
        config.queue.sort((a, b) => {
          if (a.priority !== b.priority) {
            return a.priority === 'high' ? -1 : 1;
          }
          return a.timestamp - b.timestamp;
        });
        
        // Try to process queued requests
        const processedIndexes = [];
        
        for (let i = 0; i < config.queue.length; i++) {
          const request = config.queue[i];
          const bucket = config.buckets[request.endpoint];
          
          this.refillBucket(bucket);
          
          if (bucket.tokens >= 1) {
            bucket.tokens--;
            request.resolve({ allowed: true, queued: false });
            processedIndexes.push(i);
            this.stats.queuedRequests--;
          }
        }
        
        // Remove processed requests
        for (let i = processedIndexes.length - 1; i >= 0; i--) {
          config.queue.splice(processedIndexes[i], 1);
        }
      }
    }, 100); // Check every 100ms
  }
  
  /**
   * Get current rate limit status
   */
  getStatus(provider = null) {
    if (provider) {
      const config = this.providers[provider];
      if (!config) return null;
      
      const status = {
        provider,
        endpoints: {},
        queueLength: config.queue.length
      };
      
      for (const [endpoint, bucket] of Object.entries(config.buckets)) {
        this.refillBucket(bucket);
        status.endpoints[endpoint] = {
          available: Math.floor(bucket.tokens),
          max: bucket.maxTokens,
          percentUsed: ((bucket.maxTokens - bucket.tokens) / bucket.maxTokens) * 100,
          resetIn: Math.ceil((bucket.maxTokens - bucket.tokens) / bucket.refillRate)
        };
      }
      
      return status;
    }
    
    // Return all providers status
    const allStatus = {
      stats: this.stats,
      providers: {}
    };
    
    for (const providerName of Object.keys(this.providers)) {
      allStatus.providers[providerName] = this.getStatus(providerName);
    }
    
    return allStatus;
  }
  
  /**
   * Emergency reset - clear all limits (use cautiously)
   */
  emergencyReset(provider = null) {
    console.warn(`[RateLimiter] Emergency reset triggered for ${provider || 'all providers'}`);
    
    if (provider) {
      const config = this.providers[provider];
      if (config) {
        // Reset buckets
        for (const bucket of Object.values(config.buckets)) {
          bucket.tokens = bucket.maxTokens;
          bucket.lastRefill = Date.now();
        }
        // Clear queue
        config.queue = [];
      }
    } else {
      // Reset all providers
      this.initializeBuckets();
      for (const config of Object.values(this.providers)) {
        config.queue = [];
      }
    }
  }
  
  /**
   * Wrap an API call with rate limiting
   */
  async executeWithLimit(provider, endpoint, apiCall, priority = 'normal') {
    const limitCheck = await this.checkLimit(provider, endpoint, priority);
    
    if (limitCheck.allowed) {
      // Execute immediately
      try {
        return await apiCall();
      } catch (error) {
        // If we get a rate limit error from the API, update our tracking
        if (error.response?.status === 429 || error.message?.includes('rate limit')) {
          console.error(`[RateLimiter] Unexpected rate limit from ${provider}/${endpoint}`);
          // Reduce available tokens to prevent further hits
          const bucket = this.providers[provider]?.buckets[endpoint];
          if (bucket) {
            bucket.tokens = Math.max(0, bucket.tokens - 5);
          }
        }
        throw error;
      }
    } else if (limitCheck.queued) {
      // Wait for queue processing
      await limitCheck.promise;
      // Try to execute after being dequeued
      return apiCall();
    } else {
      // Request blocked
      throw new Error(limitCheck.reason);
    }
  }
  
  /**
   * Spread out scheduled calls to avoid bursts
   */
  scheduleSpread(calls, intervalMs = 1000) {
    const results = [];
    
    calls.forEach((call, index) => {
      const delay = index * intervalMs;
      const promise = new Promise((resolve) => {
        setTimeout(async () => {
          try {
            const result = await call();
            resolve({ success: true, result });
          } catch (error) {
            resolve({ success: false, error });
          }
        }, delay);
      });
      
      results.push(promise);
    });
    
    return Promise.all(results);
  }
}

// Singleton instance
module.exports = new RateLimiter();
