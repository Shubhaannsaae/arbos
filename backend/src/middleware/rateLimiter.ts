import { Request, Response, NextFunction } from 'express';
import rateLimit, { RateLimitRequestHandler } from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import { createClient } from 'redis';
import { logger } from '../utils/logger';
import { SubscriptionTier } from '../models/User';

interface RateLimitOptions {
  windowMs: number;
  max: number;
  message?: string;
  standardHeaders?: boolean;
  legacyHeaders?: boolean;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
  skip?: (req: Request) => boolean;
  onLimitReached?: (req: Request, res: Response) => void;
}

interface TierLimits {
  requests: number;
  windowMs: number;
  burstLimit?: number;
}

class RateLimiterService {
  private redisClient: any;
  private redisStore: any;
  private isRedisConnected: boolean = false;

  // Rate limit configurations for different subscription tiers
  private tierLimits: Record<SubscriptionTier, TierLimits> = {
    [SubscriptionTier.FREE]: {
      requests: 100,
      windowMs: 60 * 60 * 1000, // 1 hour
      burstLimit: 20
    },
    [SubscriptionTier.BASIC]: {
      requests: 1000,
      windowMs: 60 * 60 * 1000, // 1 hour
      burstLimit: 50
    },
    [SubscriptionTier.PREMIUM]: {
      requests: 10000,
      windowMs: 60 * 60 * 1000, // 1 hour
      burstLimit: 200
    },
    [SubscriptionTier.ENTERPRISE]: {
      requests: 100000,
      windowMs: 60 * 60 * 1000, // 1 hour
      burstLimit: 1000
    }
  };

  // Special limits for sensitive operations
  private operationLimits = {
    login: {
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 5, // 5 attempts per 15 minutes
      message: 'Too many login attempts, please try again later.'
    },
    createAgent: {
      windowMs: 60 * 60 * 1000, // 1 hour
      max: 10, // 10 agents per hour
      message: 'Too many agent creation attempts, please try again later.'
    },
    executeArbitrage: {
      windowMs: 60 * 1000, // 1 minute
      max: 30, // 30 executions per minute
      message: 'Too many arbitrage executions, please slow down.'
    },
    rebalancePortfolio: {
      windowMs: 5 * 60 * 1000, // 5 minutes
      max: 5, // 5 rebalances per 5 minutes
      message: 'Too many portfolio rebalances, please wait before trying again.'
    },
    securityAction: {
      windowMs: 60 * 1000, // 1 minute
      max: 10, // 10 security actions per minute
      message: 'Too many security actions, please slow down.'
    }
  };

  constructor() {
    this.initializeRedis();
  }

  /**
   * Initialize Redis connection for distributed rate limiting
   */
  private async initializeRedis(): Promise<void> {
    try {
      if (process.env.REDIS_URL) {
        this.redisClient = createClient({
          url: process.env.REDIS_URL,
          socket: {
            connectTimeout: 10000,
            lazyConnect: true
          }
        });

        this.redisClient.on('error', (error: Error) => {
          logger.error('Redis client error:', error);
          this.isRedisConnected = false;
        });

        this.redisClient.on('connect', () => {
          logger.info('Redis client connected for rate limiting');
          this.isRedisConnected = true;
        });

        this.redisClient.on('disconnect', () => {
          logger.warn('Redis client disconnected');
          this.isRedisConnected = false;
        });

        await this.redisClient.connect();

        // Initialize Redis store for rate limiting
        this.redisStore = new RedisStore({
          sendCommand: (...args: string[]) => this.redisClient.sendCommand(args),
          prefix: 'rl:arbos:',
        });

        logger.info('Redis rate limiting store initialized');
      } else {
        logger.warn('Redis URL not provided, using memory store for rate limiting');
      }
    } catch (error) {
      logger.error('Failed to initialize Redis for rate limiting:', error);
      this.isRedisConnected = false;
    }
  }

  /**
   * Create a general rate limiter
   */
  public createRateLimiter = (options: RateLimitOptions): RateLimitRequestHandler => {
    const limiterOptions = {
      windowMs: options.windowMs,
      max: options.max,
      message: {
        success: false,
        error: 'RateLimitExceeded',
        message: options.message || 'Too many requests, please try again later.',
        retryAfter: Math.ceil(options.windowMs / 1000)
      },
      standardHeaders: options.standardHeaders !== false,
      legacyHeaders: options.legacyHeaders !== false,
      skipSuccessfulRequests: options.skipSuccessfulRequests || false,
      skipFailedRequests: options.skipFailedRequests || false,
      keyGenerator: options.keyGenerator || this.defaultKeyGenerator,
      skip: options.skip || (() => false),
      onLimitReached: (req: Request, res: Response) => {
        logger.warn('Rate limit exceeded', {
          ip: req.ip,
          userId: req.user?.id,
          userAgent: req.headers['user-agent'],
          path: req.path,
          method: req.method
        });

        if (options.onLimitReached) {
          options.onLimitReached(req, res);
        }
      },
      store: this.isRedisConnected ? this.redisStore : undefined
    };

    return rateLimit(limiterOptions);
  };

  /**
   * Default key generator for rate limiting
   */
  private defaultKeyGenerator = (req: Request): string => {
    // Prefer user ID for authenticated requests, fall back to IP
    if (req.user?.id) {
      return `user:${req.user.id}`;
    }
    
    // Use API key if present
    if (req.headers['x-api-key']) {
      return `apikey:${req.headers['x-api-key']}`;
    }
    
    // Fall back to IP address
    return `ip:${req.ip}`;
  };

  /**
   * Subscription tier-based rate limiter
   */
  public tierBasedLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      windowMs: 60 * 60 * 1000, // 1 hour base window
      max: (req: Request) => {
        const tier = req.user?.subscriptionTier || SubscriptionTier.FREE;
        return this.tierLimits[tier].requests;
      },
      keyGenerator: (req: Request) => {
        const tier = req.user?.subscriptionTier || SubscriptionTier.FREE;
        return `${this.defaultKeyGenerator(req)}:tier:${tier}`;
      },
      message: 'Rate limit exceeded for your subscription tier. Please upgrade or try again later.'
    });
  };

  /**
   * Burst rate limiter for short-term limits
   */
  public burstLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      windowMs: 60 * 1000, // 1 minute window
      max: (req: Request) => {
        const tier = req.user?.subscriptionTier || SubscriptionTier.FREE;
        return this.tierLimits[tier].burstLimit || 10;
      },
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:burst`;
      },
      message: 'Too many requests in a short time. Please slow down.'
    });
  };

  /**
   * Login attempt rate limiter
   */
  public loginLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      ...this.operationLimits.login,
      keyGenerator: (req: Request) => {
        // Use email/wallet address if provided, otherwise IP
        const identifier = req.body.email || req.body.walletAddress || req.ip;
        return `login:${identifier}`;
      },
      onLimitReached: (req: Request, res: Response) => {
        logger.warn('Login rate limit exceeded', {
          ip: req.ip,
          identifier: req.body.email || req.body.walletAddress,
          userAgent: req.headers['user-agent']
        });
      }
    });
  };

  /**
   * Agent operation rate limiter
   */
  public agentOperationLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      ...this.operationLimits.createAgent,
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:agent`;
      }
    });
  };

  /**
   * Arbitrage execution rate limiter
   */
  public arbitrageLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      ...this.operationLimits.executeArbitrage,
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:arbitrage`;
      },
      skip: (req: Request) => {
        // Skip for premium and enterprise users
        const tier = req.user?.subscriptionTier;
        return tier === SubscriptionTier.PREMIUM || tier === SubscriptionTier.ENTERPRISE;
      }
    });
  };

  /**
   * Portfolio rebalancing rate limiter
   */
  public rebalanceLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      ...this.operationLimits.rebalancePortfolio,
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:rebalance`;
      }
    });
  };

  /**
   * Security action rate limiter
   */
  public securityLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      ...this.operationLimits.securityAction,
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:security`;
      }
    });
  };

  /**
   * API endpoint specific rate limiter
   */
  public apiEndpointLimiter = (endpoint: string, customLimits?: Partial<RateLimitOptions>): RateLimitRequestHandler => {
    return this.createRateLimiter({
      windowMs: 60 * 1000, // 1 minute default
      max: 60, // 60 requests per minute default
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:api:${endpoint}`;
      },
      message: `Too many requests to ${endpoint} endpoint. Please try again later.`,
      ...customLimits
    });
  };

  /**
   * Dynamic rate limiter based on user behavior
   */
  public dynamicLimiter = (): RateLimitRequestHandler => {
    return this.createRateLimiter({
      windowMs: 60 * 1000, // 1 minute
      max: async (req: Request) => {
        const userId = req.user?.id;
        if (!userId) return 10; // Anonymous users get lower limits

        try {
          // Get user's recent behavior metrics
          const behaviorScore = await this.getUserBehaviorScore(userId);
          const tier = req.user?.subscriptionTier || SubscriptionTier.FREE;
          const baseLimits = this.tierLimits[tier];
          
          // Adjust limits based on behavior score
          // Higher behavior score = more trustworthy = higher limits
          const adjustment = Math.max(0.5, Math.min(2.0, behaviorScore));
          const adjustedLimit = Math.floor((baseLimits.burstLimit || 10) * adjustment);
          
          return adjustedLimit;
        } catch (error) {
          logger.error('Error calculating dynamic rate limit:', error);
          return 10; // Fallback limit
        }
      },
      keyGenerator: (req: Request) => {
        return `${this.defaultKeyGenerator(req)}:dynamic`;
      }
    });
  };

  /**
   * Get user behavior score for dynamic rate limiting
   */
  private async getUserBehaviorScore(userId: string): Promise<number> {
    try {
      // Implementation would analyze user's historical behavior
      // Factors: successful transactions, failed attempts, account age, etc.
      // For now, return a default score
      return 1.0; // Neutral score
    } catch (error) {
      logger.error('Error getting user behavior score:', error);
      return 0.5; // Conservative score on error
    }
  }

  /**
   * Whitelist middleware to bypass rate limiting
   */
  public whitelist = (whitelistedIPs: string[] = []): ((req: Request, res: Response, next: NextFunction) => void) => {
    const defaultWhitelist = [
      '127.0.0.1',
      '::1',
      'localhost'
    ];

    const combinedWhitelist = [...defaultWhitelist, ...whitelistedIPs];

    return (req: Request, res: Response, next: NextFunction): void => {
      const clientIP = req.ip;
      
      if (combinedWhitelist.includes(clientIP)) {
        // Skip rate limiting for whitelisted IPs
        req.rateLimit = {
          limit: Infinity,
          used: 0,
          remaining: Infinity,
          reset: new Date()
        };
        return next();
      }

      next();
    };
  };

  /**
   * Rate limit status middleware
   */
  public rateLimitStatus = () => {
    return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
      try {
        if (this.isRedisConnected && req.user?.id) {
          const key = `user:${req.user.id}`;
          const tier = req.user.subscriptionTier;
          const limits = this.tierLimits[tier];
          
          // Get current usage from Redis
          const usage = await this.getCurrentUsage(key);
          
          // Add rate limit info to response headers
          res.setHeader('X-RateLimit-Limit', limits.requests);
          res.setHeader('X-RateLimit-Remaining', Math.max(0, limits.requests - usage));
          res.setHeader('X-RateLimit-Reset', new Date(Date.now() + limits.windowMs).toISOString());
          res.setHeader('X-RateLimit-Window', Math.floor(limits.windowMs / 1000));
        }
        
        next();
      } catch (error) {
        logger.error('Error getting rate limit status:', error);
        next();
      }
    };
  };

  /**
   * Get current usage from Redis
   */
  private async getCurrentUsage(key: string): Promise<number> {
    try {
      if (!this.isRedisConnected) return 0;
      
      const usage = await this.redisClient.get(`rl:arbos:${key}`);
      return parseInt(usage) || 0;
    } catch (error) {
      logger.error('Error getting current usage:', error);
      return 0;
    }
  }

  /**
   * Reset rate limit for a specific key
   */
  public async resetRateLimit(key: string): Promise<boolean> {
    try {
      if (!this.isRedisConnected) return false;
      
      await this.redisClient.del(`rl:arbos:${key}`);
      logger.info(`Rate limit reset for key: ${key}`);
      return true;
    } catch (error) {
      logger.error('Error resetting rate limit:', error);
      return false;
    }
  }

  /**
   * Get rate limiting stats
   */
  public async getRateLimitStats(): Promise<any> {
    try {
      if (!this.isRedisConnected) {
        return { error: 'Redis not connected' };
      }

      const keys = await this.redisClient.keys('rl:arbos:*');
      const stats = {
        totalKeys: keys.length,
        activeUsers: keys.filter(key => key.includes('user:')).length,
        activeIPs: keys.filter(key => key.includes('ip:')).length,
        redisConnected: this.isRedisConnected
      };

      return stats;
    } catch (error) {
      logger.error('Error getting rate limit stats:', error);
      return { error: 'Failed to get stats' };
    }
  }
}

// Create singleton instance
export const rateLimiterService = new RateLimiterService();

// Export convenience functions
export const createRateLimiter = rateLimiterService.createRateLimiter;
export const tierBasedLimiter = rateLimiterService.tierBasedLimiter;
export const burstLimiter = rateLimiterService.burstLimiter;
export const loginLimiter = rateLimiterService.loginLimiter;
export const agentOperationLimiter = rateLimiterService.agentOperationLimiter;
export const arbitrageLimiter = rateLimiterService.arbitrageLimiter;
export const rebalanceLimiter = rateLimiterService.rebalanceLimiter;
export const securityLimiter = rateLimiterService.securityLimiter;
export const apiEndpointLimiter = rateLimiterService.apiEndpointLimiter;
export const dynamicLimiter = rateLimiterService.dynamicLimiter;
export const whitelist = rateLimiterService.whitelist;
export const rateLimitStatus = rateLimiterService.rateLimitStatus;

export default rateLimiterService;
