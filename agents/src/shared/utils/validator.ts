import Joi from 'joi';
import { ethers } from 'ethers';

// Custom validation functions
export const customValidators = {
  ethereumAddress: (value: string, helpers: any) => {
    if (!ethers.isAddress(value)) {
      return helpers.error('ethereum.address');
    }
    return value.toLowerCase();
  },

  transactionHash: (value: string, helpers: any) => {
    if (!/^0x[a-fA-F0-9]{64}$/.test(value)) {
      return helpers.error('ethereum.transactionHash');
    }
    return value.toLowerCase();
  },

  bigNumber: (value: any, helpers: any) => {
    try {
      if (typeof value === 'string' || typeof value === 'number') {
        return ethers.BigNumber.from(value);
      }
      if (ethers.BigNumber.isBigNumber(value)) {
        return value;
      }
      return helpers.error('bigNumber.invalid');
    } catch {
      return helpers.error('bigNumber.invalid');
    }
  },

  chainId: (value: number, helpers: any) => {
    const supportedChains = [1, 137, 42161, 43114, 56, 250];
    if (!supportedChains.includes(value)) {
      return helpers.error('chainId.unsupported');
    }
    return value;
  },

  percentage: (value: number, helpers: any) => {
    if (value < 0 || value > 100) {
      return helpers.error('percentage.range');
    }
    return value;
  },

  riskScore: (value: number, helpers: any) => {
    if (value < 0 || value > 100) {
      return helpers.error('riskScore.range');
    }
    return value;
  },

  confidence: (value: number, helpers: any) => {
    if (value < 0 || value > 1) {
      return helpers.error('confidence.range');
    }
    return value;
  }
};

// Extend Joi with custom validators
const extendedJoi = Joi.extend(
  {
    type: 'ethereum',
    base: Joi.string(),
    messages: {
      'ethereum.address': '{{#label}} must be a valid Ethereum address',
      'ethereum.transactionHash': '{{#label}} must be a valid transaction hash'
    },
    rules: {
      address: {
        method: customValidators.ethereumAddress
      },
      transactionHash: {
        method: customValidators.transactionHash
      }
    }
  },
  {
    type: 'bigNumber',
    base: Joi.any(),
    messages: {
      'bigNumber.invalid': '{{#label}} must be a valid BigNumber'
    },
    validate: customValidators.bigNumber
  },
  {
    type: 'chainId',
    base: Joi.number(),
    messages: {
      'chainId.unsupported': '{{#label}} must be a supported chain ID'
    },
    validate: customValidators.chainId
  },
  {
    type: 'percentage',
    base: Joi.number(),
    messages: {
      'percentage.range': '{{#label}} must be between 0 and 100'
    },
    validate: customValidators.percentage
  },
  {
    type: 'riskScore',
    base: Joi.number(),
    messages: {
      'riskScore.range': '{{#label}} must be between 0 and 100'
    },
    validate: customValidators.riskScore
  },
  {
    type: 'confidence',
    base: Joi.number(),
    messages: {
      'confidence.range': '{{#label}} must be between 0 and 1'
    },
    validate: customValidators.confidence
  }
);

// Common schemas
export const schemas = {
  agentContext: extendedJoi.object({
    agentId: Joi.string().uuid().required(),
    agentType: Joi.string().valid('arbitrage', 'portfolio', 'yield', 'security', 'orchestrator').required(),
    userId: Joi.string().uuid().required(),
    sessionId: Joi.string().uuid().required(),
    networkIds: Joi.array().items(extendedJoi.chainId()).min(1).required(),
    timestamp: Joi.number().integer().positive().required(),
    gasPrice: extendedJoi.bigNumber().required(),
    nonce: Joi.number().integer().min(0).required()
  }),

  agentDecision: extendedJoi.object({
    action: Joi.string().required(),
    confidence: extendedJoi.confidence().required(),
    reasoning: Joi.string().min(10).required(),
    parameters: Joi.object().required(),
    riskScore: extendedJoi.riskScore().required(),
    expectedOutcome: Joi.any().required(),
    alternatives: Joi.array().items(
      Joi.object({
        action: Joi.string().required(),
        probability: extendedJoi.confidence().required(),
        outcome: Joi.any().required()
      })
    ).optional()
  }),

  tokenInfo: extendedJoi.object({
    address: extendedJoi.ethereum().address().required(),
    symbol: Joi.string().min(1).max(20).required(),
    name: Joi.string().min(1).max(100).required(),
    decimals: Joi.number().integer().min(0).max(30).required(),
    chainId: extendedJoi.chainId().required(),
    logoURI: Joi.string().uri().optional(),
    tags: Joi.array().items(Joi.string()).optional(),
    isStable: Joi.boolean().required(),
    isNative: Joi.boolean().required()
  }),

  priceData: extendedJoi.object({
    token: extendedJoi.ethereum().address().required(),
    price: extendedJoi.bigNumber().required(),
    priceUsd: Joi.number().positive().required(),
    timestamp: Joi.number().integer().positive().required(),
    source: Joi.string().required(),
    confidence: extendedJoi.confidence().required(),
    volume24h: extendedJoi.bigNumber().required(),
    change24h: Joi.number().required(),
    marketCap: extendedJoi.bigNumber().optional(),
    circulatingSupply: extendedJoi.bigNumber().optional(),
    totalSupply: extendedJoi.bigNumber().optional()
  }),

  arbitrageOpportunity: extendedJoi.object({
    id: Joi.string().uuid().required(),
    tokenPair: Joi.string().pattern(/^[A-Z]+\/[A-Z]+$/).required(),
    sourceExchange: Joi.object({
      name: Joi.string().required(),
      address: extendedJoi.ethereum().address().required(),
      chainId: extendedJoi.chainId().required(),
      price: extendedJoi.bigNumber().required(),
      liquidity: extendedJoi.bigNumber().required(),
      fee: Joi.number().min(0).max(10).required()
    }).required(),
    targetExchange: Joi.object({
      name: Joi.string().required(),
      address: extendedJoi.ethereum().address().required(),
      chainId: extendedJoi.chainId().required(),
      price: extendedJoi.bigNumber().required(),
      liquidity: extendedJoi.bigNumber().required(),
      fee: Joi.number().min(0).max(10).required()
    }).required(),
    priceDifference: extendedJoi.bigNumber().required(),
    priceDifferencePercentage: Joi.number().required(),
    potentialProfit: extendedJoi.bigNumber().required(),
    potentialProfitPercentage: Joi.number().positive().required(),
    maxTradeSize: extendedJoi.bigNumber().required(),
    estimatedGasCost: extendedJoi.bigNumber().required(),
    totalCosts: extendedJoi.bigNumber().required(),
    netProfit: extendedJoi.bigNumber().required(),
    confidence: extendedJoi.confidence().required(),
    riskScore: extendedJoi.riskScore().required(),
    executionComplexity: Joi.string().valid('simple', 'medium', 'complex').required(),
    requiredCapital: extendedJoi.bigNumber().required(),
    estimatedExecutionTime: Joi.number().positive().required(),
    detectedAt: Joi.number().integer().positive().required(),
    expiresAt: Joi.number().integer().positive().required()
  }),

  portfolioPosition: extendedJoi.object({
    token: schemas.tokenInfo,
    amount: extendedJoi.bigNumber().required(),
    value: extendedJoi.bigNumber().required(),
    valueUsd: Joi.number().positive().required(),
    percentage: extendedJoi.percentage().required(),
    averageCost: extendedJoi.bigNumber().required(),
    unrealizedPnl: extendedJoi.bigNumber().required(),
    unrealizedPnlPercentage: Joi.number().required(),
    lastUpdated: Joi.number().integer().positive().required()
  }),

  yieldOpportunity: extendedJoi.object({
    id: Joi.string().uuid().required(),
    protocol: Joi.string().required(),
    strategy: Joi.string().valid('lending', 'liquidity_mining', 'staking', 'farming').required(),
    baseApr: Joi.number().min(0).max(1000).required(),
    rewardApr: Joi.number().min(0).max(1000).required(),
    totalApr: Joi.number().min(0).max(1000).required(),
    apy: Joi.number().min(0).max(1000).required(),
    tvl: extendedJoi.bigNumber().required(),
    minimumDeposit: extendedJoi.bigNumber().required(),
    maximumDeposit: extendedJoi.bigNumber().optional(),
    lockupPeriod: Joi.number().integer().min(0).required(),
    withdrawalFee: extendedJoi.percentage().required(),
    performanceFee: extendedJoi.percentage().required(),
    impermanentLossRisk: extendedJoi.riskScore().required(),
    protocolRisk: extendedJoi.riskScore().required(),
    auditScore: extendedJoi.riskScore().required(),
    sustainability: extendedJoi.riskScore().required()
  }),

  transactionRequest: extendedJoi.object({
    to: extendedJoi.ethereum().address().required(),
    value: extendedJoi.bigNumber().default(ethers.BigNumber.from(0)),
    data: Joi.string().pattern(/^0x[a-fA-F0-9]*$/).default('0x'),
    gasLimit: extendedJoi.bigNumber().required(),
    gasPrice: extendedJoi.bigNumber().optional(),
    maxFeePerGas: extendedJoi.bigNumber().optional(),
    maxPriorityFeePerGas: extendedJoi.bigNumber().optional(),
    nonce: Joi.number().integer().min(0).optional(),
    type: Joi.number().valid(0, 1, 2).default(2)
  }),

  agentConfiguration: extendedJoi.object({
    enabled: Joi.boolean().required(),
    executionInterval: Joi.number().integer().min(1000).required(),
    maxConcurrentExecutions: Joi.number().integer().min(1).max(10).required(),
    riskTolerance: Joi.string().valid('low', 'medium', 'high').required(),
    constraints: Joi.object({
      maxDailyTransactions: Joi.number().integer().min(1).required(),
      maxDailyVolume: extendedJoi.bigNumber().required(),
      maxPositionSize: extendedJoi.bigNumber().required(),
      minProfitThreshold: Joi.number().min(0).max(100).required(),
      maxSlippage: Joi.number().min(0).max(20).required(),
      maxGasPrice: extendedJoi.bigNumber().required(),
      allowedTokens: Joi.array().items(extendedJoi.ethereum().address()).required(),
      blockedTokens: Joi.array().items(extendedJoi.ethereum().address()).optional(),
      allowedProtocols: Joi.array().items(Joi.string()).required(),
      blockedProtocols: Joi.array().items(Joi.string()).optional()
    }).required()
  })
};

// Validation functions
export class Validator {
  static validate<T>(data: any, schema: Joi.Schema): { value: T; error?: string } {
    const { error, value } = schema.validate(data, {
      abortEarly: false,
      stripUnknown: true,
      convert: true
    });

    if (error) {
      return {
        value: data,
        error: error.details.map(detail => detail.message).join('; ')
      };
    }

    return { value };
  }

  static validateAgentContext(data: any) {
    return this.validate(data, schemas.agentContext);
  }

  static validateAgentDecision(data: any) {
    return this.validate(data, schemas.agentDecision);
  }

  static validateTokenInfo(data: any) {
    return this.validate(data, schemas.tokenInfo);
  }

  static validatePriceData(data: any) {
    return this.validate(data, schemas.priceData);
  }

  static validateArbitrageOpportunity(data: any) {
    return this.validate(data, schemas.arbitrageOpportunity);
  }

  static validatePortfolioPosition(data: any) {
    return this.validate(data, schemas.portfolioPosition);
  }

  static validateYieldOpportunity(data: any) {
    return this.validate(data, schemas.yieldOpportunity);
  }

  static validateTransactionRequest(data: any) {
    return this.validate(data, schemas.transactionRequest);
  }

  static validateAgentConfiguration(data: any) {
    return this.validate(data, schemas.agentConfiguration);
  }

  static isValidEthereumAddress(address: string): boolean {
    return ethers.isAddress(address);
  }

  static isValidTransactionHash(hash: string): boolean {
    return /^0x[a-fA-F0-9]{64}$/.test(hash);
  }

  static isValidBigNumber(value: any): boolean {
    try {
      ethers.BigNumber.from(value);
      return true;
    } catch {
      return false;
    }
  }

  static sanitizeString(input: string, maxLength: number = 1000): string {
    return input
      .trim()
      .replace(/[<>]/g, '') // Remove potential HTML tags
      .substring(0, maxLength);
  }

  static sanitizeNumeric(input: any): number | null {
    const num = Number(input);
    return isNaN(num) || !isFinite(num) ? null : num;
  }

  static validateRange(value: number, min: number, max: number): boolean {
    return value >= min && value <= max;
  }

  static validateArrayItems<T>(
    array: any[],
    itemValidator: (item: any) => { value: T; error?: string }
  ): { values: T[]; errors: string[] } {
    const values: T[] = [];
    const errors: string[] = [];

    array.forEach((item, index) => {
      const result = itemValidator(item);
      if (result.error) {
        errors.push(`Item ${index}: ${result.error}`);
      } else {
        values.push(result.value);
      }
    });

    return { values, errors };
  }
}

export default Validator;
