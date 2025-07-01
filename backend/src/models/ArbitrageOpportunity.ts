export interface ArbitrageOpportunity {
  id: string;
  userId: string;
  agentId: string;
  tokenPair: string;
  sourceExchange: ExchangeInfo;
  targetExchange: ExchangeInfo;
  sourceChain: ChainInfo;
  targetChain: ChainInfo;
  priceDifference: number;
  priceDifferencePercentage: number;
  potentialProfit: number;
  potentialProfitUSD: number;
  estimatedGasCost: number;
  estimatedGasCostUSD: number;
  netProfit: number;
  netProfitUSD: number;
  minTradeSize: number;
  maxTradeSize: number;
  liquidityAvailable: number;
  priceImpact: number;
  slippage: number;
  confidence: number;
  riskScore: number;
  timeToExpiry: number;
  detectedAt: Date;
  executedAt?: Date;
  status: OpportunityStatus;
  executionTxHash?: string;
  actualProfit?: number;
  actualProfitUSD?: number;
  failureReason?: string;
  metadata: OpportunityMetadata;
}

export interface ExchangeInfo {
  name: string;
  address: string;
  type: ExchangeType;
  price: number;
  liquidity: number;
  volume24h: number;
  fees: ExchangeFees;
  slippage: number;
  apiEndpoint?: string;
}

export interface ChainInfo {
  chainId: number;
  name: string;
  rpcUrl: string;
  gasPrice: number;
  blockTime: number;
  confirmations: number;
  bridgeAddress?: string;
  bridgeFee?: number;
}

export interface ExchangeFees {
  trading: number;
  gas: number;
  bridge?: number;
  protocol: number;
}

export interface OpportunityMetadata {
  detectionMethod: DetectionMethod;
  marketConditions: MarketConditions;
  historicalSuccess: number;
  competitionLevel: number;
  urgency: UrgencyLevel;
  tags: string[];
  relatedOpportunities?: string[];
  blockNumber: string;
  timestamp: Date;
}

export interface MarketConditions {
  volatility: number;
  volume: number;
  trend: MarketTrend;
  sentiment: MarketSentiment;
  liquidityIndex: number;
  fear_greed_index: number;
}

export enum OpportunityStatus {
  DETECTED = 'detected',
  ANALYZING = 'analyzing',
  APPROVED = 'approved',
  EXECUTING = 'executing',
  EXECUTED = 'executed',
  FAILED = 'failed',
  EXPIRED = 'expired',
  CANCELLED = 'cancelled'
}

export enum ExchangeType {
  DEX = 'dex',
  CEX = 'cex',
  AMM = 'amm',
  ORDER_BOOK = 'order_book',
  AGGREGATOR = 'aggregator'
}

export enum DetectionMethod {
  PRICE_FEED = 'price_feed',
  WEBSOCKET = 'websocket',
  API_POLLING = 'api_polling',
  BLOCKCHAIN_EVENTS = 'blockchain_events',
  ML_PREDICTION = 'ml_prediction'
}

export enum MarketTrend {
  BULLISH = 'bullish',
  BEARISH = 'bearish',
  SIDEWAYS = 'sideways',
  VOLATILE = 'volatile'
}

export enum MarketSentiment {
  VERY_BULLISH = 'very_bullish',
  BULLISH = 'bullish',
  NEUTRAL = 'neutral',
  BEARISH = 'bearish',
  VERY_BEARISH = 'very_bearish'
}

export enum UrgencyLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface CreateArbitrageOpportunityDto {
  tokenPair: string;
  sourceExchange: ExchangeInfo;
  targetExchange: ExchangeInfo;
  sourceChain: ChainInfo;
  targetChain: ChainInfo;
  priceDifference: number;
  potentialProfit: number;
  estimatedGasCost: number;
  metadata: OpportunityMetadata;
}

export interface ArbitrageFilter {
  minProfit?: number;
  maxRiskScore?: number;
  chains?: number[];
  exchanges?: string[];
  tokens?: string[];
  status?: OpportunityStatus[];
  timeRange?: {
    start: Date;
    end: Date;
  };
}

export interface ArbitrageAnalytics {
  totalOpportunities: number;
  successfulExecutions: number;
  totalProfit: number;
  averageProfit: number;
  successRate: number;
  averageExecutionTime: number;
  topTokenPairs: Array<{
    pair: string;
    count: number;
    profit: number;
  }>;
  topExchanges: Array<{
    exchange: string;
    count: number;
    profit: number;
  }>;
  dailyStats: Array<{
    date: Date;
    opportunities: number;
    executions: number;
    profit: number;
  }>;
}
