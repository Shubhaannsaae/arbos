import { Request, Response, NextFunction } from 'express';
import { z, ZodError, ZodSchema } from 'zod';
import { logger } from '../utils/logger';

// Common validation schemas
export const commonSchemas = {
  // Ethereum address validation
  ethereumAddress: z.string().regex(/^0x[a-fA-F0-9]{40}$/, 'Invalid Ethereum address'),
  
  // Transaction hash validation
  transactionHash: z.string().regex(/^0x[a-fA-F0-9]{64}$/, 'Invalid transaction hash'),
  
  // Chain ID validation
  chainId: z.number().int().positive(),
  
  // Pagination parameters
  pagination: z.object({
    page: z.number().int().min(1).default(1),
    limit: z.number().int().min(1).max(100).default(10),
    sortBy: z.string().optional(),
    sortOrder: z.enum(['asc', 'desc']).default('desc')
  }),
  
  // Date range validation
  dateRange: z.object({
    start: z.string().datetime().optional(),
    end: z.string().datetime().optional()
  }).refine(data => {
    if (data.start && data.end) {
      return new Date(data.start) <= new Date(data.end);
    }
    return true;
  }, 'Start date must be before end date'),
  
  // Amount validation (for financial amounts)
  amount: z.number().positive().finite(),
  
  // Percentage validation
  percentage: z.number().min(0).max(100),
  
  // Slippage validation
  slippage: z.number().min(0).max(50), // 0-50%
  
  // Gas limit validation
  gasLimit: z.number().int().min(21000).max(10000000),
  
  // UUID validation
  uuid: z.string().uuid(),
  
  // Risk tolerance validation
  riskTolerance: z.enum(['conservative', 'moderate', 'aggressive', 'very_aggressive']),
  
  // Subscription tier validation
  subscriptionTier: z.enum(['free', 'basic', 'premium', 'enterprise'])
};

// User validation schemas
export const userSchemas = {
  createUser: z.object({
    walletAddress: commonSchemas.ethereumAddress,
    email: z.string().email().optional(),
    riskTolerance: commonSchemas.riskTolerance.optional(),
    preferences: z.object({
      notifications: z.object({
        email: z.boolean().default(true),
        push: z.boolean().default(true),
        sms: z.boolean().default(false),
        arbitrageAlerts: z.boolean().default(true),
        portfolioRebalancing: z.boolean().default(true),
        securityAlerts: z.boolean().default(true)
      }).optional(),
      trading: z.object({
        maxSlippage: z.number().min(0).max(10).default(0.5),
        autoRebalance: z.boolean().default(false),
        rebalanceThreshold: z.number().min(1).max(50).default(5),
        preferredDexes: z.array(z.string()).default([]),
        blacklistedTokens: z.array(commonSchemas.ethereumAddress).default([])
      }).optional(),
      risk: z.object({
        maxPositionSize: z.number().min(0).max(100).default(20),
        stopLossPercentage: z.number().min(0).max(50).default(10),
        takeProfitPercentage: z.number().min(0).max(200).default(50),
        allowLeverage: z.boolean().default(false),
        maxLeverage: z.number().min(1).max(10).default(2)
      }).optional()
    }).optional()
  }),

  updateUser: z.object({
    email: z.string().email().optional(),
    riskTolerance: commonSchemas.riskTolerance.optional(),
    preferences: z.object({
      notifications: z.object({
        email: z.boolean(),
        push: z.boolean(),
        sms: z.boolean(),
        arbitrageAlerts: z.boolean(),
        portfolioRebalancing: z.boolean(),
        securityAlerts: z.boolean()
      }).partial(),
      trading: z.object({
        maxSlippage: z.number().min(0).max(10),
        autoRebalance: z.boolean(),
        rebalanceThreshold: z.number().min(1).max(50),
        preferredDexes: z.array(z.string()),
        blacklistedTokens: z.array(commonSchemas.ethereumAddress)
      }).partial(),
      risk: z.object({
        maxPositionSize: z.number().min(0).max(100),
        stopLossPercentage: z.number().min(0).max(50),
        takeProfitPercentage: z.number().min(0).max(200),
        allowLeverage: z.boolean(),
        maxLeverage: z.number().min(1).max(10)
      }).partial()
    }).partial().optional(),
    subscriptionTier: commonSchemas.subscriptionTier.optional()
  })
};

// Portfolio validation schemas
export const portfolioSchemas = {
  createPortfolio: z.object({
    name: z.string().min(3).max(50),
    description: z.string().max(500).optional(),
    targetAllocation: z.array(z.object({
      tokenAddress: commonSchemas.ethereumAddress,
      symbol: z.string().min(1).max(10),
      name: z.string().min(1).max(50),
      percentage: commonSchemas.percentage,
      decimals: z.number().int().min(0).max(18).default(18),
      priceUSD: z.number().nonnegative().default(0)
    })).min(1).max(20),
    rebalanceSettings: z.object({
      enabled: z.boolean(),
      trigger: z.enum(['threshold', 'time', 'volatility', 'market_conditions']),
      threshold: z.number().min(0.1).max(50),
      frequency: z.enum(['daily', 'weekly', 'monthly', 'quarterly']).optional(),
      slippageTolerance: z.number().min(0.1).max(10),
      gasOptimization: z.boolean(),
      minTradeSize: z.number().positive()
    }),
    restrictions: z.object({
      maxTokens: z.number().int().min(1).max(50),
      minTokenPercentage: z.number().min(0).max(100),
      maxTokenPercentage: z.number().min(0).max(100),
      allowedTokens: z.array(commonSchemas.ethereumAddress).optional(),
      blockedTokens: z.array(commonSchemas.ethereumAddress).optional(),
      allowedProtocols: z.array(z.string()).optional(),
      blockedProtocols: z.array(z.string()).optional()
    }).optional(),
    chainId: commonSchemas.chainId
  }).refine(data => {
    // Validate allocation percentages sum to 100%
    const totalPercentage = data.targetAllocation.reduce((sum, allocation) => sum + allocation.percentage, 0);
    return Math.abs(totalPercentage - 100) < 0.01;
  }, 'Target allocation percentages must sum to 100%'),

  updatePortfolio: z.object({
    name: z.string().min(3).max(50).optional(),
    description: z.string().max(500).optional(),
    targetAllocation: z.array(z.object({
      tokenAddress: commonSchemas.ethereumAddress,
      symbol: z.string().min(1).max(10),
      name: z.string().min(1).max(50),
      percentage: commonSchemas.percentage,
      decimals: z.number().int().min(0).max(18),
      priceUSD: z.number().nonnegative()
    })).min(1).max(20).optional(),
    rebalanceSettings: z.object({
      enabled: z.boolean(),
      trigger: z.enum(['threshold', 'time', 'volatility', 'market_conditions']),
      threshold: z.number().min(0.1).max(50),
      frequency: z.enum(['daily', 'weekly', 'monthly', 'quarterly']).optional(),
      slippageTolerance: z.number().min(0.1).max(10),
      gasOptimization: z.boolean(),
      minTradeSize: z.number().positive()
    }).optional(),
    restrictions: z.object({
      maxTokens: z.number().int().min(1).max(50),
      minTokenPercentage: z.number().min(0).max(100),
      maxTokenPercentage: z.number().min(0).max(100),
      allowedTokens: z.array(commonSchemas.ethereumAddress).optional(),
      blockedTokens: z.array(commonSchemas.ethereumAddress).optional(),
      allowedProtocols: z.array(z.string()).optional(),
      blockedProtocols: z.array(z.string()).optional()
    }).optional()
  }),

  rebalancePortfolio: z.object({
    strategy: z.enum(['optimal', 'aggressive', 'conservative']).default('optimal'),
    maxSlippage: commonSchemas.slippage.default(0.5),
    gasOptimization: z.boolean().default(true),
    gasLimit: commonSchemas.gasLimit.optional()
  })
};

// Agent validation schemas
export const agentSchemas = {
  createAgent: z.object({
    name: z.string().min(3).max(50),
    type: z.enum(['arbitrage', 'portfolio', 'yield', 'security', 'orchestrator']),
    configuration: z.object({
      model: z.enum(['gpt-4', 'gpt-4-turbo', 'claude-3', 'gemini-pro', 'custom']),
      parameters: z.object({
        riskTolerance: z.number().min(0).max(100),
        maxPositionSize: z.number().min(0).max(100),
        slippageTolerance: commonSchemas.slippage,
        gasOptimization: z.boolean(),
        frequencyMs: z.number().int().min(1000).max(86400000), // 1 second to 24 hours
        timeoutMs: z.number().int().min(5000).max(300000), // 5 seconds to 5 minutes
        retryAttempts: z.number().int().min(0).max(10),
        confidenceThreshold: z.number().min(0).max(1),
        profitThreshold: z.number().min(0).max(100),
        stopLossThreshold: z.number().min(0).max(50)
      }),
      constraints: z.object({
        maxDailyTransactions: z.number().int().min(0).max(10000),
        maxDailyVolume: z.number().min(0),
        allowedTokens: z.array(commonSchemas.ethereumAddress).optional(),
        blockedTokens: z.array(commonSchemas.ethereumAddress).optional(),
        allowedProtocols: z.array(z.string()).optional(),
        blockedProtocols: z.array(z.string()).optional(),
        allowedChains: z.array(commonSchemas.chainId),
        tradingHours: z.object({
          enabled: z.boolean(),
          timezone: z.string(),
          startTime: z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/),
          endTime: z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/),
          weekends: z.boolean()
        }).optional()
      }),
      tools: z.array(z.string()),
      notifications: z.object({
        onError: z.boolean(),
        onSuccess: z.boolean(),
        onThreshold: z.boolean(),
        channels: z.array(z.enum(['email', 'sms', 'push', 'webhook', 'slack', 'discord'])),
        webhook: z.string().url().optional()
      })
    }),
    permissions: z.object({
      canTrade: z.boolean(),
      canRebalance: z.boolean(),
      canWithdraw: z.boolean(),
      canApprove: z.boolean(),
      canCrossChain: z.boolean(),
      maxTransactionValue: z.number().min(0),
      allowedContracts: z.array(commonSchemas.ethereumAddress).optional(),
      emergencyStop: z.boolean()
    })
  }),

  updateAgent: z.object({
    name: z.string().min(3).max(50).optional(),
    configuration: z.object({
      parameters: z.object({
        riskTolerance: z.number().min(0).max(100),
        maxPositionSize: z.number().min(0).max(100),
        slippageTolerance: commonSchemas.slippage,
        gasOptimization: z.boolean(),
        frequencyMs: z.number().int().min(1000).max(86400000),
        timeoutMs: z.number().int().min(5000).max(300000),
        retryAttempts: z.number().int().min(0).max(10),
        confidenceThreshold: z.number().min(0).max(1),
        profitThreshold: z.number().min(0).max(100),
        stopLossThreshold: z.number().min(0).max(50)
      }).partial(),
      constraints: z.object({
        maxDailyTransactions: z.number().int().min(0).max(10000),
        maxDailyVolume: z.number().min(0),
        allowedTokens: z.array(commonSchemas.ethereumAddress),
        blockedTokens: z.array(commonSchemas.ethereumAddress),
        allowedProtocols: z.array(z.string()),
        blockedProtocols: z.array(z.string()),
        allowedChains: z.array(commonSchemas.chainId),
        tradingHours: z.object({
          enabled: z.boolean(),
          timezone: z.string(),
          startTime: z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/),
          endTime: z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/),
          weekends: z.boolean()
        })
      }).partial(),
      notifications: z.object({
        onError: z.boolean(),
        onSuccess: z.boolean(),
        onThreshold: z.boolean(),
        channels: z.array(z.enum(['email', 'sms', 'push', 'webhook', 'slack', 'discord'])),
        webhook: z.string().url()
      }).partial()
    }).partial().optional(),
    permissions: z.object({
      canTrade: z.boolean(),
      canRebalance: z.boolean(),
      canWithdraw: z.boolean(),
      canApprove: z.boolean(),
      canCrossChain: z.boolean(),
      maxTransactionValue: z.number().min(0),
      allowedContracts: z.array(commonSchemas.ethereumAddress),
      emergencyStop: z.boolean()
    }).partial().optional(),
    isEnabled: z.boolean().optional()
  }),

  executeAction: z.object({
    action: z.string().min(1),
    parameters: z.record(z.any()).optional()
  })
};

// Arbitrage validation schemas
export const arbitrageSchemas = {
  executeArbitrage: z.object({
    opportunityId: commonSchemas.uuid,
    amount: commonSchemas.amount,
    maxSlippage: commonSchemas.slippage.default(0.5),
    gasLimit: commonSchemas.gasLimit.optional()
  }),

  createOpportunity: z.object({
    tokenPair: z.string().min(1),
    sourceExchange: z.object({
      name: z.string(),
      address: commonSchemas.ethereumAddress,
      type: z.enum(['dex', 'cex', 'amm', 'order_book', 'aggregator']),
      price: z.number().positive(),
      liquidity: z.number().nonnegative(),
      volume24h: z.number().nonnegative(),
      fees: z.object({
        trading: z.number().min(0).max(10),
        gas: z.number().nonnegative(),
        bridge: z.number().nonnegative().optional(),
        protocol: z.number().nonnegative()
      }),
      slippage: z.number().min(0).max(50)
    }),
    targetExchange: z.object({
      name: z.string(),
      address: commonSchemas.ethereumAddress,
      type: z.enum(['dex', 'cex', 'amm', 'order_book', 'aggregator']),
      price: z.number().positive(),
      liquidity: z.number().nonnegative(),
      volume24h: z.number().nonnegative(),
      fees: z.object({
        trading: z.number().min(0).max(10),
        gas: z.number().nonnegative(),
        bridge: z.number().nonnegative().optional(),
        protocol: z.number().nonnegative()
      }),
      slippage: z.number().min(0).max(50)
    }),
    sourceChain: z.object({
      chainId: commonSchemas.chainId,
      name: z.string(),
      rpcUrl: z.string().url(),
      gasPrice: z.number().positive(),
      blockTime: z.number().positive(),
      confirmations: z.number().int().positive()
    }),
    targetChain: z.object({
      chainId: commonSchemas.chainId,
      name: z.string(),
      rpcUrl: z.string().url(),
      gasPrice: z.number().positive(),
      blockTime: z.number().positive(),
      confirmations: z.number().int().positive()
    }),
    priceDifference: z.number().positive(),
    potentialProfit: z.number().positive(),
    estimatedGasCost: z.number().nonnegative(),
    metadata: z.object({
      detectionMethod: z.enum(['price_feed', 'websocket', 'api_polling', 'blockchain_events', 'ml_prediction']),
      marketConditions: z.object({
        volatility: z.number().min(0).max(1),
        volume: z.number().nonnegative(),
        trend: z.enum(['bullish', 'bearish', 'sideways', 'volatile']),
        sentiment: z.enum(['very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish']),
        liquidityIndex: z.number().min(0).max(1),
        fear_greed_index: z.number().min(0).max(100)
      }),
      historicalSuccess: z.number().min(0).max(1),
      competitionLevel: z.number().min(0).max(1),
      urgency: z.enum(['low', 'medium', 'high', 'critical']),
      tags: z.array(z.string()),
      blockNumber: z.string(),
      timestamp: z.string().datetime()
    })
  })
};

// Security validation schemas
export const securitySchemas = {
  analyzeTransaction: z.object({
    txHash: commonSchemas.transactionHash,
    chainId: commonSchemas.chainId
  }),

  monitorWallet: z.object({
    walletAddress: commonSchemas.ethereumAddress,
    chainId: commonSchemas.chainId,
    monitoringLevel: z.enum(['basic', 'standard', 'enhanced']).default('standard')
  }),

  reportIncident: z.object({
    type: z.enum([
      'suspicious_transaction',
      'unusual_activity',
      'high_risk_contract',
      'price_manipulation',
      'front_running',
      'mev_attack',
      'phishing_attempt',
      'wallet_compromise',
      'smart_contract_vulnerability'
    ]),
    severity: z.enum(['low', 'medium', 'high', 'critical']),
    description: z.string().min(10).max(1000),
    txHash: commonSchemas.transactionHash.optional(),
    chainId: commonSchemas.chainId.optional(),
    metadata: z.record(z.any()).optional()
  }),

  emergencyStop: z.object({
    reason: z.string().min(10).max(500),
    scope: z.enum(['all', 'agents', 'portfolios', 'arbitrage']).default('all')
  })
};

/**
 * Generic validation middleware factory
 */
export function validateData<T extends ZodSchema>(schema: T) {
  return (req: Request, res: Response, next: NextFunction): void => {
    try {
      const validatedData = schema.parse(req.body);
      req.body = validatedData;
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const errorMessages = error.errors.map((issue) => ({
          field: issue.path.join('.'),
          message: issue.message,
          code: issue.code,
          received: issue.received
        }));

        logger.warn('Request validation failed', {
          path: req.path,
          method: req.method,
          errors: errorMessages,
          userId: req.user?.id
        });

        res.status(400).json({
          success: false,
          error: 'ValidationError',
          message: 'Invalid request data',
          details: errorMessages
        });
      } else {
        logger.error('Unexpected validation error:', error);
        res.status(500).json({
          success: false,
          error: 'InternalServerError',
          message: 'Validation processing error'
        });
      }
    }
  };
}

/**
 * Validate query parameters
 */
export function validateQuery<T extends ZodSchema>(schema: T) {
  return (req: Request, res: Response, next: NextFunction): void => {
    try {
      const validatedQuery = schema.parse(req.query);
      req.query = validatedQuery;
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const errorMessages = error.errors.map((issue) => ({
          field: issue.path.join('.'),
          message: issue.message,
          code: issue.code,
          received: issue.received
        }));

        res.status(400).json({
          success: false,
          error: 'QueryValidationError',
          message: 'Invalid query parameters',
          details: errorMessages
        });
      } else {
        res.status(500).json({
          success: false,
          error: 'InternalServerError',
          message: 'Query validation processing error'
        });
      }
    }
  };
}

/**
 * Validate URL parameters
 */
export function validateParams<T extends ZodSchema>(schema: T) {
  return (req: Request, res: Response, next: NextFunction): void => {
    try {
      const validatedParams = schema.parse(req.params);
      req.params = validatedParams;
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const errorMessages = error.errors.map((issue) => ({
          field: issue.path.join('.'),
          message: issue.message,
          code: issue.code,
          received: issue.received
        }));

        res.status(400).json({
          success: false,
          error: 'ParamValidationError',
          message: 'Invalid URL parameters',
          details: errorMessages
        });
      } else {
        res.status(500).json({
          success: false,
          error: 'InternalServerError',
          message: 'Parameter validation processing error'
        });
      }
    }
  };
}

/**
 * Validate request headers
 */
export function validateHeaders<T extends ZodSchema>(schema: T) {
  return (req: Request, res: Response, next: NextFunction): void => {
    try {
      const validatedHeaders = schema.parse(req.headers);
      // Don't overwrite headers, just validate them
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const errorMessages = error.errors.map((issue) => ({
          field: issue.path.join('.'),
          message: issue.message,
          code: issue.code,
          received: issue.received
        }));

        res.status(400).json({
          success: false,
          error: 'HeaderValidationError',
          message: 'Invalid request headers',
          details: errorMessages
        });
      } else {
        res.status(500).json({
          success: false,
          error: 'InternalServerError',
          message: 'Header validation processing error'
        });
      }
    }
  };
}

// Convenience exports
export const validate = {
  body: validateData,
  query: validateQuery,
  params: validateParams,
  headers: validateHeaders
};

export default validate;
