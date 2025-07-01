import { BigNumber } from 'ethers';

export interface TokenInfo {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  chainId: number;
  logoURI?: string;
  tags: string[];
  isStable: boolean;
  isNative: boolean;
}

export interface PriceData {
  token: string;
  price: BigNumber;
  priceUsd: number;
  timestamp: number;
  source: string;
  confidence: number;
  volume24h: BigNumber;
  change24h: number;
  marketCap?: BigNumber;
  circulatingSupply?: BigNumber;
  totalSupply?: BigNumber;
}

export interface LiquidityPool {
  address: string;
  protocol: string;
  version: string;
  chainId: number;
  tokens: TokenInfo[];
  reserves: BigNumber[];
  totalSupply: BigNumber;
  fee: number;
  volume24h: BigNumber;
  volumeWeek: BigNumber;
  tvl: BigNumber;
  apr: number;
  apy: number;
  createdAt: number;
  lastUpdated: number;
}

export interface TradingPair {
  baseToken: TokenInfo;
  quoteToken: TokenInfo;
  pools: LiquidityPool[];
  bestBid: BigNumber;
  bestAsk: BigNumber;
  spread: number;
  volume24h: BigNumber;
  priceImpact1k: number;
  priceImpact10k: number;
  priceImpact100k: number;
}

export interface ArbitrageOpportunity {
  id: string;
  tokenPair: string;
  sourceExchange: {
    name: string;
    address: string;
    chainId: number;
    price: BigNumber;
    liquidity: BigNumber;
    fee: number;
  };
  targetExchange: {
    name: string;
    address: string;
    chainId: number;
    price: BigNumber;
    liquidity: BigNumber;
    fee: number;
  };
  priceDifference: BigNumber;
  priceDifferencePercentage: number;
  potentialProfit: BigNumber;
  potentialProfitPercentage: number;
  maxTradeSize: BigNumber;
  estimatedGasCost: BigNumber;
  totalCosts: BigNumber;
  netProfit: BigNumber;
  confidence: number;
  riskScore: number;
  executionComplexity: 'simple' | 'medium' | 'complex';
  requiredCapital: BigNumber;
  estimatedExecutionTime: number;
  detectedAt: number;
  expiresAt: number;
  chainlinkData: {
    sourcePrice: BigNumber;
    targetPrice: BigNumber;
    priceAge: number;
    feedReliability: number;
  };
}

export interface YieldOpportunity {
  id: string;
  protocol: string;
  pool: LiquidityPool;
  strategy: 'lending' | 'liquidity_mining' | 'staking' | 'farming';
  baseApr: number;
  rewardApr: number;
  totalApr: number;
  apy: number;
  tvl: BigNumber;
  userTvl: BigNumber;
  minimumDeposit: BigNumber;
  maximumDeposit: BigNumber;
  lockupPeriod: number;
  withdrawalFee: number;
  performanceFee: number;
  impermanentLossRisk: number;
  protocolRisk: number;
  auditScore: number;
  sustainability: number;
  tokenEmissions: {
    token: TokenInfo;
    rate: BigNumber;
    duration: number;
    vestingPeriod: number;
  }[];
  requirements: {
    minimumBalance: BigNumber;
    whitelistRequired: boolean;
    kycRequired: boolean;
    geographicRestrictions: string[];
  };
}

export interface MarketConditions {
  overall: 'bullish' | 'bearish' | 'neutral' | 'volatile';
  volatilityIndex: number;
  fearGreedIndex: number;
  liquidityConditions: 'abundant' | 'normal' | 'tight' | 'stressed';
  gasPrice: {
    current: BigNumber;
    average24h: BigNumber;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  networkCongestion: {
    ethereum: number;
    polygon: number;
    arbitrum: number;
    avalanche: number;
    bsc: number;
  };
  correlations: {
    btcEth: number;
    ethAlt: number;
    usdStables: number;
  };
  sentiment: {
    social: number;
    onChain: number;
    technical: number;
    fundamental: number;
  };
  risks: {
    systemicRisk: number;
    liquidityRisk: number;
    volatilityRisk: number;
    correlationRisk: number;
    concentrationRisk: number;
  };
}

export interface TradingSignal {
  id: string;
  type: 'buy' | 'sell' | 'hold';
  strength: number;
  confidence: number;
  timeframe: string;
  asset: TokenInfo;
  currentPrice: BigNumber;
  targetPrice: BigNumber;
  stopLoss: BigNumber;
  reasoning: string;
  indicators: Record<string, number>;
  riskReward: number;
  probability: number;
  validUntil: number;
  source: string;
}

export interface Portfolio {
  id: string;
  userId: string;
  name: string;
  description: string;
  totalValue: BigNumber;
  totalValueUsd: number;
  positions: PortfolioPosition[];
  allocation: AllocationTarget[];
  performance: PortfolioPerformance;
  riskMetrics: PortfolioRiskMetrics;
  rebalancing: RebalancingConfig;
  constraints: PortfolioConstraints;
}

export interface PortfolioPosition {
  token: TokenInfo;
  amount: BigNumber;
  value: BigNumber;
  valueUsd: number;
  percentage: number;
  averageCost: BigNumber;
  unrealizedPnl: BigNumber;
  unrealizedPnlPercentage: number;
  lastUpdated: number;
}

export interface AllocationTarget {
  token: TokenInfo;
  targetPercentage: number;
  minPercentage: number;
  maxPercentage: number;
  rebalanceThreshold: number;
}

export interface PortfolioPerformance {
  totalReturn: BigNumber;
  totalReturnPercentage: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  trackingError: number;
}

export interface PortfolioRiskMetrics {
  overallRiskScore: number;
  concentrationRisk: number;
  liquidityRisk: number;
  volatilityRisk: number;
  correlationRisk: number;
  valueAtRisk: {
    day1: BigNumber;
    week1: BigNumber;
    month1: BigNumber;
  };
  expectedShortfall: {
    day1: BigNumber;
    week1: BigNumber;
    month1: BigNumber;
  };
  riskContributions: Array<{
    token: TokenInfo;
    contribution: number;
  }>;
}

export interface RebalancingConfig {
  enabled: boolean;
  strategy: 'threshold' | 'periodic' | 'volatility' | 'momentum';
  frequency: number;
  threshold: number;
  maxSlippage: number;
  minTradeSize: BigNumber;
  gasOptimization: boolean;
  timeConstraints: {
    startHour: number;
    endHour: number;
    weekendsEnabled: boolean;
    blackoutPeriods: Array<{
      start: number;
      end: number;
      reason: string;
    }>;
  };
}

export interface PortfolioConstraints {
  maxPositions: number;
  minPositionSize: BigNumber;
  maxPositionSize: BigNumber;
  maxSectorExposure: number;
  maxSingleAssetExposure: number;
  allowedAssets: string[];
  blockedAssets: string[];
  leverage: {
    enabled: boolean;
    maxLeverage: number;
    maintenanceMargin: number;
  };
  riskLimits: {
    maxDailyLoss: BigNumber;
    maxDrawdown: number;
    maxVolatility: number;
    maxBeta: number;
  };
}
